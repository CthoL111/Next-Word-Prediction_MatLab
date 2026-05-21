% PART 2: Build Trigram Model (optional extra credit)
% Counts how often word C follows the pair "word A word B"
function model = buildTrigram(tokens)
    model.trigram = containers.Map();  % "w1 w2" -> next word counts

    for i = 1:numel(tokens)-2
        key = [tokens{i} ' ' tokens{i+1}];
        w3  = tokens{i+2};

        if ~isKey(model.trigram, key)
            model.trigram(key) = containers.Map();
        end
        inner = model.trigram(key);
        if isKey(inner, w3)
            inner(w3) = inner(w3) + 1;
        else
            inner(w3) = 1;
        end
        model.trigram(key) = inner;
    end
end