% PART 5: Simple UI for Next Word Prediction
function launchUI(bigramModel, trigramModel, coMatrix, vocab)
    fig = uifigure('Name', 'Next Word Predictor', 'Position', [200 200 520 360]);

    uilabel(fig, 'Text', 'Enter word(s):', 'Position', [20 300 120 22]);
    inputField = uieditfield(fig, 'text', 'Position', [150 300 240 22], ...
        'Placeholder', 'e.g. cat  or  the cat');

    uilabel(fig, 'Text', 'Model:', 'Position', [20 260 60 22]);
    modeDD = uidropdown(fig, ...
        'Items', {'Bigram','Trigram','Vector Similarity'}, ...
        'Position', [90 260 180 22]);

    uibutton(fig, 'Text', 'Predict', ...
        'Position', [290 258 100 30], ...
        'ButtonPushedFcn', @(~,~) doPrediction());

    uilabel(fig, 'Text', 'Results:', 'Position', [20 225 80 20]);
    resultArea = uitextarea(fig, 'Position', [20 60 480 160], 'Editable', 'off');

    function doPrediction()
        try
            inputText   = strtrim(lower(inputField.Value));
            words       = strsplit(inputText);
            words       = words(~cellfun('isempty', words));
            modelChoice = modeDD.Value;
            result      = '';

            switch modelChoice

                case 'Bigram'
                    % Uses last word to predict next word
                    w1 = words{end};
                    if ~isKey(bigramModel.bigram, w1)
                        result = sprintf('Word "%s" not found in model.\nTry: cat, dog, fish, bird, the, a', w1);
                    else
                        inner  = bigramModel.bigram(w1);
                        ws     = keys(inner);
                        counts = cell2mat(values(inner));
                        total  = sum(counts);
                        [~, idx] = sort(counts, 'descend');
                        topK   = min(5, numel(ws));
                        result = sprintf('Top predictions after "%s":\n\n', w1);
                        for k = 1:topK
                            p = counts(idx(k)) / total;
                            result = [result sprintf('  %d.  %-15s  probability = %.4f\n', k, ws{idx(k)}, p)]; %#ok
                        end
                    end

                case 'Trigram'
                    % Uses last 2 words to predict next word
                    if numel(words) < 2
                        result = 'Please enter at least 2 words for Trigram.';
                    else
                        w1  = words{end-1};
                        w2  = words{end};
                        key = [w1 ' ' w2];
                        if ~isKey(trigramModel.trigram, key)
                            result = sprintf('Pair "%s %s" not found.\nTry: the cat, a fish, the bird', w1, w2);
                        else
                            inner  = trigramModel.trigram(key);
                            ws     = keys(inner);
                            counts = cell2mat(values(inner));
                            [~, idx] = sort(counts, 'descend');
                            topK   = min(5, numel(ws));
                            result = sprintf('Top predictions after "%s %s":\n\n', w1, w2);
                            for k = 1:topK
                                result = [result sprintf('  %d.  %s\n', k, ws{idx(k)})]; %#ok
                            end
                        end
                    end

                case 'Vector Similarity'
                    % Finds words with similar context using cosine similarity
                    w1        = words{end};
                    wordIndex = containers.Map(vocab, 1:numel(vocab));
                    if ~isKey(wordIndex, w1)
                        result = sprintf('Word "%s" not found in vocabulary.', w1);
                    else
                        widx = wordIndex(w1);
                        vec  = coMatrix(widx, :);
                        sims = zeros(1, numel(vocab));
                        for i = 1:numel(vocab)
                            other = coMatrix(i, :);
                            denom = norm(vec) * norm(other);
                            if denom > 0
                                sims(i) = dot(vec, other) / denom;
                            end
                        end
                        sims(widx) = -inf;
                        [sortedSims, sortedIdx] = sort(sims, 'descend');
                        topK   = min(5, numel(vocab));
                        result = sprintf('Words most similar to "%s":\n\n', w1);
                        for k = 1:topK
                            result = [result sprintf('  %d.  %-15s  similarity = %.4f\n', k, vocab{sortedIdx(k)}, sortedSims(k))]; %#ok
                        end
                    end
            end

            resultArea.Value = result;

        catch ME
            resultArea.Value = ['ERROR: ' ME.message];
        end
    end
end
