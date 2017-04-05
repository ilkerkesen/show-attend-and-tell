# LSTM model - input * weight, concatenated weights
function lstm(weight, bias, hidden, cell, input, ctx)
    gates   = hcat(input,hidden,ctx) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function logprob(output, ypred, mask=nothing)
    nrows,ncols = size(ypred)
    index = similar(output)
    @inbounds for i=1:length(output)
        index[i] = i + (output[i]-1)*nrows
    end
    o1 = logp(ypred,2)
    index = mask == nothing ? index : index[mask]
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

# attention mechanism
function att(w,a,h, o=Dict())
    # get sizes
    B,L,D  = size(a); H = size(h,2)
    attdrop = get(o, :attdrop, 0.0)

    # flatten region features
    aflat = reshape(a, B*L, D)

    # visual features projection
    aatt = aflat*w["wce"] # size(wce): (D,D), projection
    aatt = reshape(aatt, B, L, D)

    # hidden states projection
    hatt = h * w["whe"]
    hatt = reshape(hatt, B, 1, D) # size(whe): (H,D)

    # et calculation
    et = tanh(aatt .+ hatt)
    et = reshape(et, B*L, D)
    et = dropout(et, attdrop) * w["watt"] # watt: D,1
    et = reshape(et, B, L)

    # alpha = logp(et, 2)
    et1 = maximum(et, 2)
    et2 = et .- et1
    et3 = exp(et2)
    et4 = sum(et3, 2)
    et5 = et3 ./ et4

    alpha = exp(et5)
    alpha = alpha ./ sum(alpha,2)
    alpha = reshape(alpha, size(alpha)..., 1)

    context = alpha .* a # B,L,1 * B,L,D
    context = sum(context,2)
    context = reshape(context, size(context,1), size(context,3))
end
