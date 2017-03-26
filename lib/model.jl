# loss functions
function loss(w, s, visual, captions, masks; o=Dict(), values=[])
    finetune = get(o, :finetune, false)
    visual = convert(o[:atype], visual)
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
    atype = o[:atype]

    # set dropouts
    vembdrop = get(o, :vembdrop, 0.0)
    wembdrop = get(o, :wembdrop, 0.0)
    softdrop = get(o, :softdrop, 0.0)
    fc7drop  = get(o, :fc7drop, 0.0)

    # init state
    h = mean(vis, 2)
    h = reshape(vis, size(h,1), size(h,3))
    s = (h,h)

    # textual features
    for t = 1:length(seq)-1
        # make input
        x = w["wemb"][seq[t],:]
        x = dropout(x, wembdrop)

        # get context vector
        ctx = att(w,a,s[1],o)

        # lstm
        (s[1], s[2]) = lstm(w["wdec"], w["bdec"], s[1], s[2], x, ctx)

        # prediction
        ht = dropout(s[1], softdrop)
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
    atype = typeof(AutoGrad.getval(w[1]))<:KnetArray?KnetArray:Array
    if wcnn != nothing
        vis = KnetArray(vis)
        vis = vgg19(wcnn, vis)
        vis = transpose(vis)
    else
        vis = convert(atype, vis)
    end
    x = vis * w[5]
    (s[1], s[2]) = lstm(w[1], w[2], s[1], s[2], x)

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
            onehotvec = zeros(Cuchar, 1, vocab.size)
            onehotvec[word2index(vocab, word)] = 1
            x = convert(atype, onehotvec) * w[6]
            x = w[:]
            (st[1], st[2]) = lstm(w[1], w[2], st[1], st[2], x, ctx)
            ypred = logp(st[1] * w[3] .+ w[4], 2)
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
