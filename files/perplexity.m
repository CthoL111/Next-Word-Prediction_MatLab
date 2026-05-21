% PART 4: Calculate Perplexity (lower = better model)
% Uses Laplace (add-1) smoothing for unseen word pairs
function pp = perplexity(model, testTokens)
    logProb = 0;
    N = numel(testTokens) - 1;
    V = numel(keys(model.unigram));  % vocabulary size

    for i = 1:N
        w1 = testTokens{i};
        w2 = testTokens{i+1};

        if isKey(model.bigram, w1)
            inner      = model.bigram(w1);
            count_pair = 0;
            if isKey(inner, w2), count_pair = inner(w2); end
            count_w1   = model.unigram(w1);
            p = (count_pair + 1) / (count_w1 + V);  % Laplace smoothing
        else
            p = 1 / V;
        end

        logProb = logProb + log2(p);
    end

    pp = 2^(-logProb / N);
    fprintf('Perplexity      : %.4f\n', pp);
end
