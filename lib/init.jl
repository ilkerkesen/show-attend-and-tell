# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2)
    state[1] = zeros(batchsize, hidden)
    state[2] = zeros(batchsize, hidden)
    return map(s->convert(atype,s), state)
end

# initialize all weights of decoder network
function initweights(o::Dict)
    w = Dict()
    L,D = o[:visual]
    w["wdec"] = o[:winit]*randn(o[:embed]+o[:hidden]+D, 4*o[:hidden])
    w["bdec"] = zeros(1, 4*o[:hidden])
    w["bdec"][1:o[:hidden]] = 1 # forget gate bias
    w["wsoft"] = o[:winit]*randn(o[:hidden], o[:vocabsize])
    w["bsoft"] = zeros(1, o[:vocabsize])
    w["wemb"] = o[:winit]*randn(o[:vocabsize], o[:embed])
    w["wce"] = o[:winit]*randn(D, D)
    w["whe"] = o[:winit]*randn(o[:hidden], D)
    w["watt"] = o[:winit]*randn(D,1)
    return convert_weight(o[:atype], w)
end
