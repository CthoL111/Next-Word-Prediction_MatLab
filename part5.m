% ============================================================
% PART 5: Application and Documentation (15%)
% ============================================================

% --- 1. Save Model ---
function saveModel(bigramModel, trigramModel, coMatrix, vocab, filename)
    if nargin < 5
        filename = 'nwp_model.mat';
    end
    save(filename, 'bigramModel', 'trigramModel', 'coMatrix', 'vocab');
    fprintf('Model saved to: %s\n', filename);
end

% --- 2. Load Model ---
function [bigramModel, trigramModel, coMatrix, vocab] = loadModel(filename)
    if nargin < 1
        filename = 'nwp_model.mat';
    end
    data         = load(filename);
    bigramModel  = data.bigramModel;
    trigramModel = data.trigramModel;
    coMatrix     = data.coMatrix;
    vocab        = data.vocab;
    fprintf('Model loaded from: %s\n', filename);
end

% --- 3. UI ---
function launchUI(bigramModel, trigramModel, coMatrix, vocab, posMap)

    fig = uifigure('Name', 'Next Word Predictor', ...
                   'Position', [200 150 520 500]);

    % title
    uilabel(fig, ...
        'Text', 'Next Word Prediction', ...
        'Position', [20 455 480 30], ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    % input field
    uilabel(fig, 'Text', 'Type here:', ...
        'Position', [20 415 80 22], ...
        'FontWeight', 'bold');

    inputField = uieditfield(fig, 'text', ...
        'Position', [110 415 380 28], ...
        'FontSize', 13, ...
        'Placeholder', 'Start typing...', ...
        'ValueChangedFcn', @(~,~) onInputChanged());

    % model selector
    uilabel(fig, 'Text', 'Model:', ...
        'Position', [20 378 50 22], ...
        'FontWeight', 'bold');

    modeDD = uidropdown(fig, ...
        'Items', {'Bigram','Trigram','Vector Similarity'}, ...
        'Position', [80 378 150 25]);

    % instruction label
    uilabel(fig, ...
        'Text', '↑↓ to move  |  Enter to select  |  Backspace to delete', ...
        'Position', [20 350 480 20], ...
        'FontSize', 10, ...
        'FontColor', [0.5 0.5 0.5], ...
        'HorizontalAlignment', 'center');

    % suggestion list
    uilabel(fig, 'Text', 'Suggestions:', ...
        'Position', [20 322 100 20], ...
        'FontWeight', 'bold');

    suggList = uilistbox(fig, ...
        'Position', [20 180 480 140], ...
        'Items', {}, ...
        'FontSize', 13, ...
        'DoubleClickedFcn', @(~,~) onSuggestionSelected());

    % chosen word display
    uilabel(fig, 'Text', 'Chosen sentence:', ...
        'Position', [20 148 150 22], ...
        'FontWeight', 'bold');

    chosenArea = uitextarea(fig, ...
        'Position', [20 60 480 85], ...
        'Editable', 'off', ...
        'FontSize', 13);

    % clear button
    uibutton(fig, 'Text', 'Clear All', ...
        'Position', [390 20 110 30], ...
        'ButtonPushedFcn', @(~,~) clearAll());

    % status
    statusLabel = uilabel(fig, ...
        'Text', 'Start typing to see suggestions...', ...
        'Position', [20 20 360 22], ...
        'FontColor', [0.3 0.6 0.3]);

    % track chosen sentence
    chosenWords = {};

    % ---- when user types ----
    function onInputChanged()
        inputText = strtrim(lower(inputField.Value));
        if isempty(inputText)
            suggList.Items = {};
            statusLabel.Text = 'Start typing...';
            return;
        end

        words     = strsplit(inputText);
        words     = words(~cellfun('isempty', words));
        numWords  = numel(words);
        modelChoice = modeDD.Value;
        preds     = {};

        try
            switch modelChoice
                case 'Bigram'
                    w1 = words{end};
                    if isKey(bigramModel.bigram, w1)
                        inner      = bigramModel.bigram(w1);
                        ws         = keys(inner);
                        counts     = cell2mat(values(inner));
                        total      = sum(counts);
                        [~, idx]   = sort(counts, 'descend');
                        topK       = min(5, numel(ws));
                        for k = 1:topK
                            p     = counts(idx(k)) / total;
                            preds{end+1} = sprintf('%-15s  (%.2f%%)', ...
                                ws{idx(k)}, p*100); %#ok
                        end
                    end

                case 'Trigram'
                    if numWords >= 2
                        w1  = words{end-1};
                        w2  = words{end};
                        key = [w1 ' ' w2];
                        if isKey(trigramModel.trigram, key)
                            inner    = trigramModel.trigram(key);
                            ws       = keys(inner);
                            counts   = cell2mat(values(inner));
                            [~, idx] = sort(counts, 'descend');
                            topK     = min(5, numel(ws));
                            for k = 1:topK
                                preds{end+1} = ws{idx(k)}; %#ok
                            end
                        end
                    else
                        % fall back to bigram if only 1 word
                        w1 = words{end};
                        if isKey(bigramModel.bigram, w1)
                            inner      = bigramModel.bigram(w1);
                            ws         = keys(inner);
                            counts     = cell2mat(values(inner));
                            [~, idx]   = sort(counts, 'descend');
                            topK       = min(5, numel(ws));
                            for k = 1:topK
                                preds{end+1} = ws{idx(k)}; %#ok
                            end
                        end
                    end

                case 'Vector Similarity'
                    w1        = words{end};
                    wordIndex = containers.Map(vocab, 1:numel(vocab));
                    if isKey(wordIndex, w1)
                        widx = wordIndex(w1);
                        vec  = coMatrix(widx,:);
                        sims = zeros(1, numel(vocab));
                        for j = 1:numel(vocab)
                            other = coMatrix(j,:);
                            denom = norm(vec)*norm(other);
                            if denom > 0
                                sims(j) = dot(vec,other)/denom;
                            end
                        end
                        sims(widx) = -inf;
                        [~, sortedIdx] = sort(sims,'descend');
                        topK = min(5, numel(vocab));
                        for k = 1:topK
                            preds{end+1} = vocab{sortedIdx(k)}; %#ok
                        end
                    end
            end

            if isempty(preds)
                suggList.Items = {'(no suggestions)'};
                statusLabel.Text = sprintf('No predictions for "%s"', words{end});
            else
                suggList.Items = preds;
                suggList.Value = preds{1};  % select first by default
                statusLabel.Text = sprintf('%d suggestions for "%s"', numel(preds), words{end});
            end

        catch ME
            statusLabel.Text = ['Error: ' ME.message];
        end
    end

    % ---- when user clicks a suggestion ----
    function onSuggestionSelected()
        selected = suggList.Value;
        if isempty(selected) || strcmp(selected, '(no suggestions)')
            return;
        end
    
        % Extract just the word (remove probability display)
        parts = strsplit(strtrim(selected));
        word  = parts{1};
    
        % Sync with what is currently typed BEFORE appending
        inputText  = strtrim(lower(inputField.Value));
        typedWords = strsplit(inputText);
        typedWords = typedWords(~cellfun('isempty', typedWords));
    
        % Build new sentence = typed words + chosen word
        newWords  = [typedWords, {word}];
        sentence  = strjoin(newWords, ' ');
    
        % Update displays
        chosenArea.Value = sentence;         % ← Fix 2: always shows full sentence
        statusLabel.Text = sprintf('Chosen: "%s" | Sentence: %s', word, sentence);
    
        % Update input for next prediction (without triggering chosenWords confusion)
        chosenWords      = newWords;         % ← Fix 1: keep in sync manually
        inputField.Value = sentence;
        onInputChanged();
    end
    

    % ---- clear everything ----
    function clearAll()
        inputField.Value = '';
        suggList.Items   = {};
        chosenArea.Value = '';
        chosenWords      = {};
        statusLabel.Text = 'Cleared. Start typing...';
    end

end
% ============================================================
% RUN PART 5
% ============================================================
fprintf('=== PART 5: Application and Documentation ===\n\n');

% --- Save Model ---
fprintf('--- Saving Model ---\n');
saveModel(bigramModel, trigramModel, coMatrix, corpus.vocab);
fprintf('\n');

% --- Load Model (verify it works) ---
fprintf('--- Loading Model ---\n');
[bgModel, tgModel, coMat, voc] = loadModel('nwp_model.mat');
fprintf('Loaded vocab size: %d\n', numel(voc));
fprintf('\n');

% --- Results Analysis ---
fprintf('--- Results Analysis ---\n');
fprintf('Corpus size    : %d tokens\n',   numel(corpus.tokens));
fprintf('Vocabulary     : %d unique words\n', corpus.vocabSize);
fprintf('Train tokens   : %d\n',          numel(trainTok));
fprintf('Test tokens    : %d\n',          numel(testTok));
fprintf('Bigram pairs   : %d\n',          numel(keys(bigramModel.bigram)));
fprintf('Trigram pairs  : %d\n',          numel(keys(trigramModel.trigram)));
fprintf('\n');

% --- Suggestions for Improvement ---
fprintf('--- Suggestions for Improvement ---\n');
fprintf('1. Use larger corpus for better accuracy\n');
fprintf('2. Add trigram smoothing (Kneser-Ney)\n');
fprintf('3. Use Word2Vec instead of co-occurrence\n');
fprintf('4. Add 4-gram model for more context\n');
fprintf('5. Use neural network (LSTM) for predictions\n');
fprintf('\n');

% --- Launch UI ---
fprintf('--- Launching UI ---\n');
launchUI(bigramModel, trigramModel, coMatrix, corpus.vocab);