% PART 4: Evaluate Bigram Model Accuracy
% Checks how often the top prediction matches the actual next word
function acc = evaluateBigram(model, testTokens)
    correct = 0;
    total   = 0;

    for i = 1:numel(testTokens)-1
        w1     = testTokens{i};
        actual = testTokens{i+1};

        if ~isKey(model.bigram, w1)
            total = total + 1;
            continue;
        end

        inner  = model.bigram(w1);
        ws     = keys(inner);
        counts = cell2mat(values(inner));
        [~, best] = max(counts);

        if strcmp(ws{best}, actual)
            correct = correct + 1;
        end
        total = total + 1;
    end

    acc = correct / total * 100;
    fprintf('Bigram Accuracy: %.2f%%  (%d/%d correct)\n', acc, correct, total);
end
