# loss functions
function loss(w, s, visual, captions, masks; o=Dict(), values=[])
    finetune = get(o, :finetune, false)
    atype = get(o, :atype, AutoGrad.getval(typeof(w["wdec"])))
    atype = atype<:Array?Array:KnetArray
    visual = convert(atype, visual)
    if finetune
        visual = vgg19(w["wcnn"], visual; o=o)
        visual = transpose(visual)
    end

    lossval, nwords = decoder(w, s, visual, captions, masks; o=o)
    push!(values, AutoGrad.getval(lossval), AutoGrad.getval(nwords))
    return lossval/nwords
end

# loss gradient functions
lossgradient = grad(loss)

# loss function for decoder network
function decoder(w, sd, vis, seq, masks; o=Dict())
    total, count = 0, 0
    atype = get(o, :atype, AutoGrad.getval(typeof(w["wdec"])))

    # set dropouts
    vembdrop = get(o, :vembdrop, 0.0)
    wembdrop = get(o, :wembdrop, 0.0)
    softdrop = get(o, :softdrop, 0.0)
    fc7drop  = get(o, :fc7drop, 0.0)

    h = sum(vis, 2) / size(vis, 2)
    h = reshape(h, size(h,1), size(h,3))
    c = copy(h)

    # textual features
    for t = 1:length(seq)-1
        # make input
        x = w["wemb"][seq[t],:]
        x = dropout(x, wembdrop)

        # get context vector
        ctx = att(w,vis,h,o)

        # make input
        x = hcat(ctx,wemb)

        # lstm
        h,c = lstm(w["wdec"], w["bdec"], h, c, x)

        # prediction
        ht = dropout(h, softdrop)
        ypred = ht * w["wsoft"] .+ w["bsoft"]
        ygold = seq[t+1]

        # loss calculcation
        total += logprob(ygold,ypred,masks[t])
        count += length(ygold)
        x = ygold
    end

    return (-total,count)
end

# generate
function generate(w, s, vis, vocab; maxlen=20, beamsize=1)
    atype = typeof(AutoGrad.getval(w["wdec"])) <: KnetArray ? KnetArray : Array
    wcnn = get(w, "wcnn", nothing)
    vis = convert(atype, vis)
    if wcnn != nothing
        vis = vgg19(wcnn, vis; o=Dict(:featuremaps=>true))
    end
    vis = reshape(vis, 1, size(vis,1)*size(vis,2), size(vis,3))

    h = sum(vis, 2) / size(vis, 2)
    h = reshape(h, size(h,1), size(h,3))
    c = copy(h)

    # language generation with (sentence, state, probability) array
    sentences = Any[(Any[SOS],h,c,0.0)]
    while true
        changed = false
        for i = 1:beamsize
            # get current sentence
            curr = shift!(sentences)
            sentence, ht, ct, prob = curr

            # get last word
            word = sentence[end]
            if word == EOS || length(sentence) >= maxlen
                push!(sentences, curr)
                continue
            end

            # get probabilities
            x = w["wemb"][word2index(vocab,word),:]
            x = reshape(x, 1, length(x))
            ctx = att(w,vis,ht)
            (ht,ct) = lstm(w["wdec"], w["bdec"], ht, ct, x, ctx)
            ypred = ht * w["wsoft"] .+ w["bsoft"]
            ypred = logp(ypred, 2)
            ypred = convert(Array{Float32}, ypred)[:]

            # add most probable predictions to array
            maxinds = sortperm(ypred, rev=true)
            for j = 1:beamsize
                ind = maxinds[j]
                new_word = index2word(vocab, ind)
                new_sentence = copy(sentence)
                new_probability = prob + ypred[ind]
                push!(new_sentence, new_word)
                push!(sentences, (new_sentence, copy(ht), copy(ct), new_probability))
            end
            changed = true

            # skip first loop
            if word == SOS
                break
            end
        end

        orders = sortperm(map(x -> x[4], sentences), rev=true)
        sentences = sentences[orders[1:beamsize]]

        if !changed
            break
        end
    end

    sentence = first(sentences)[1]
    if sentence[end] == EOS
        pop!(sentence)
    end
    push!(sentence, ".")
    output = join(filter(word -> word != UNK, sentence[2:end]), " ")
end
