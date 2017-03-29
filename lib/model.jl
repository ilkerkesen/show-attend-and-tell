# loss functions
function loss(w, s, visual, captions, masks; o=Dict(), values=[])
    finetune = get(o, :finetune, false)
    atype = get(o, :atype, AutoGrad.getval(typeof(w["wdec"])))
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

    # init state
    h = mean(vis, 2)
    h = reshape(h, size(h,1), size(h,3))
    c = copy(h)

    # textual features
    for t = 1:length(seq)-1
        # make input
        x = w["wemb"][seq[t],:]
        x = dropout(x, wembdrop)

        # get context vector
        ctx = att(w,vis,h,o)

        # lstm
        h,c = lstm(w["wdec"], w["bdec"], h, c, x, ctx)

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
function generate(w, wcnn, s, vis, vocab; maxlen=20, beamsize=1)
    atype = typeof(AutoGrad.getval(w[1])) <: KnetArray ? KnetArray : Array
    vis = convert(atype, vis)
    if wcnn != nothing
        vis = vgg19(wcnn, vis)
        vis = transpose(vis)
    end

    # init state
    h = mean(vis, 2)
    h = reshape(h, size(h,1), size(h,3))
    s = (h,h)

    # language generation with (sentence, state, probability) array
    sentences = Any[(Any[SOS],s,0.0)]
    while true
        changed = false
        for i = 1:beamsize
            # get current sentence
            curr = shift!(sentences)
            sentence, st, prob = curr

            # get last word
            word = sentence[end]
            if word == EOS || length(sentence) >= maxlen
                push!(sentences, curr)
                continue
            end

            # get probabilities
            x = w["wemb"][word2index(vocab,word),:]
            x = reshape(x, 1, length(x))
            ctx = att(w,vis,st[1],o)
            (st[1], st[2]) = lstm(w["wdec"], w["bdec"], st[1], st[2], x, ctx)
            ypred = ht * w["wsoft"] .+ w["bsoft"]
            ypred = logp(ypred, 2)
            ypred = convert(Array{Float32}, ypred)[:]

            # add most probable predictions to array
            maxinds = sortperm(ypred, rev=true)
            for j = 1:beamsize
                ind = maxinds[j]
                new_word = index2word(vocab, ind)
                new_sentence = copy(sentence)
                new_state = copy(st)
                new_probability = prob + ypred[ind]
                push!(new_sentence, new_word)
                push!(sentences, (new_sentence, new_state, new_probability))
            end
            changed = true

            # skip first loop
            if word == SOS
                break
            end
        end

        orders = sortperm(map(x -> x[3], sentences), rev=true)
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
