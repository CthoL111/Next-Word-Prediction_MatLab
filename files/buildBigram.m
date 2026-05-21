% PART 2: Build Bigram Model
% Counts how often word B follows word A
function model = buildBigram(tokens)
    model.bigram  = containers.Map();  % word -> next word counts
    model.unigram = containers.Map();  % word -> total count

    for i = 1:numel(tokens)-1
        w1 = tokens{i};
        w2 = tokens{i+1};

        % Count w1
        if isKey(model.unigram, w1)
            model.unigram(w1) = model.unigram(w1) + 1;
        else
            model.unigram(w1) = 1;
        end

        % Count w1 -> w2
        if ~isKey(model.bigram, w1)
            model.bigram(w1) = containers.Map();
        end
        inner = model.bigram(w1);
        if isKey(inner, w2)
            inner(w2) = inner(w2) + 1;
        else
            inner(w2) = 1;
        end
        model.bigram(w1) = inner;
    end
end
