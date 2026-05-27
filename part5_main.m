run('part1.m');
run('part2.m');
run('part3.m');
run('part4.m');


% ============================================================
% PART 5: Application and Documentation (15%)
% ============================================================

% --- Save Model ---
function saveModel(bigramModel, trigramModel, coMatrix, vocab, filename)
    if nargin < 5, filename = 'nwp_model.mat'; end
    save(filename, 'bigramModel', 'trigramModel', 'coMatrix', 'vocab');
    fprintf('Model saved to  : %s\n', filename);
end

% --- Load Model ---
function [bigramModel, trigramModel, coMatrix, vocab] = loadModel(filename)
    if nargin < 1, filename = 'nwp_model.mat'; end
    data         = load(filename);
    bigramModel  = data.bigramModel;
    trigramModel = data.trigramModel;
    coMatrix     = data.coMatrix;
    vocab        = data.vocab;
    fprintf('Model loaded from: %s\n', filename);
end

% --- UI ---
function launchUI(bigramModel, trigramModel, coMatrix, vocab)

    fig = uifigure('Name', 'Next Word Predictor', ...
                   'Position', [200 150 540 320]);

    % title
    uilabel(fig, ...
        'Text', 'Next Word Prediction', ...
        'Position', [20 275 500 30], ...
        'FontSize', 16, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    % instruction
    uilabel(fig, ...
        'Text', 'Bigram: 1 word  |  Trigram: 2 words  |  Vector: any words', ...
        'Position', [20 250 500 20], ...
        'FontSize', 10, ...
        'FontColor', [0.5 0.5 0.5], ...
        'HorizontalAlignment', 'center');

    % input label
    uilabel(fig, 'Text', 'Type here:', ...
        'Position', [20 213 80 22], ...
        'FontWeight', 'bold');

    % input field
    inputField = uieditfield(fig, 'text', ...
        'Position', [110 213 410 26], ...
        'FontSize', 13, ...
        'Placeholder', 'e.g. cat  or  the cat', ...
        'ValueChangedFcn', @(~,~) onInputChanged());

    % model label
    uilabel(fig, 'Text', 'Model:', ...
        'Position', [20 175 50 22], ...
        'FontWeight', 'bold');

    % model dropdown
    modeDD = uidropdown(fig, ...
        'Items', {'Bigram','Trigram','Vector Similarity'}, ...
        'Position', [80 175 200 25], ...
        'ValueChangedFcn', @(~,~) onInputChanged());

    % clear button
    uibutton(fig, 'Text', 'Clear All', ...
        'Position', [410 175 110 26], ...
        'ButtonPushedFcn', @(~,~) clearAll());

    % suggestions label
    uilabel(fig, 'Text', 'Suggestions:', ...
        'Position', [20 148 100 20], ...
        'FontWeight', 'bold');

    % suggestion listbox — double click adds word to input and searches again
    suggList = uilistbox(fig, ...
        'Position', [20 20 500 125], ...
        'Items', {}, ...
        'FontSize', 13, ...
        'DoubleClickedFcn', @(~,~) onSuggestionSelected());

    % ---- when user types ----
    function onInputChanged()
        inputText = strtrim(lower(inputField.Value));

        if ~strcmp(inputField.Value, inputText)
            inputField.Value = inputText;
        end

        if isempty(inputText)
            suggList.Items = {};
            return;
        end

        words       = strsplit(inputText);
        words       = words(~cellfun('isempty', words));
        numWords    = numel(words);
        modelChoice = modeDD.Value;
        preds       = {};

        try
            switch modelChoice

                case 'Bigram'
                    w1 = words{end};
                    if isKey(bigramModel.bigram, w1)
                        inner    = bigramModel.bigram(w1);
                        ws       = keys(inner);
                        counts   = cell2mat(values(inner));

                        dotMask  = ~strcmp(ws, '.');
                        ws       = ws(dotMask);
                        counts   = counts(dotMask);

                        total    = sum(counts);
                        [~, idx] = sort(counts, 'descend');
                        topK     = min(5, numel(ws));
                        for k = 1:topK
                            p = counts(idx(k)) / total;
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

                            dotMask  = ~strcmp(ws, '.');
                            ws       = ws(dotMask);
                            counts   = counts(dotMask);

                            [~, idx] = sort(counts, 'descend');
                            topK     = min(5, numel(ws));
                            for k = 1:topK
                                preds{end+1} = ws{idx(k)}; %#ok
                            end
                        end
                    else
                        w1 = words{end};
                        if isKey(bigramModel.bigram, w1)
                            inner    = bigramModel.bigram(w1);
                            ws       = keys(inner);
                            counts   = cell2mat(values(inner));

                            dotMask  = ~strcmp(ws, '.');
                            ws       = ws(dotMask);
                            counts   = counts(dotMask);

                            [~, idx] = sort(counts, 'descend');
                            topK     = min(5, numel(ws));
                            for k = 1:topK
                                preds{end+1} = ws{idx(k)}; %#ok
                            end
                        end
                    end

                case 'Vector Similarity'
                    wordIndex  = containers.Map(vocab, 1:numel(vocab));
                    combined   = zeros(1, numel(vocab));
                    validCount = 0;

                    for wi = 1:numel(words)
                        if isKey(wordIndex, words{wi})
                            widx     = wordIndex(words{wi});
                            combined = combined + coMatrix(widx,:);
                            validCount = validCount + 1;
                        end
                    end

                    if validCount > 0
                        combined = combined / validCount;
                        sims     = zeros(1, numel(vocab));

                        for j = 1:numel(vocab)
                            other = coMatrix(j,:);
                            denom = norm(combined) * norm(other);
                            if denom > 0
                                sims(j) = dot(combined, other) / denom;
                            end
                        end

                        for wi = 1:numel(words)
                            if isKey(wordIndex, words{wi})
                                sims(wordIndex(words{wi})) = -inf;
                            end
                        end

                        [sortedSims, sortedIdx] = sort(sims, 'descend');

                        dotMask    = ~strcmp(vocab(sortedIdx), '.');
                        sortedIdx  = sortedIdx(dotMask);
                        sortedSims = sortedSims(dotMask);

                        topK = min(5, numel(vocab));
                        for k = 1:topK
                            preds{end+1} = sprintf('%-15s  (sim=%.4f)', ...
                                vocab{sortedIdx(k)}, sortedSims(k)); %#ok
                        end
                    end
            end

            if isempty(preds)
                suggList.Items = {'(no suggestions found)'};
            else
                suggList.Items = preds;
            end

        catch ME
            suggList.Items = {['Error: ' ME.message]};
        end
    end

    % ---- double click: append word to input then search again ----
    function onSuggestionSelected()
        selected = suggList.Value;
        if isempty(selected) || strcmp(selected, '(no suggestions found)')
            return;
        end

        % extract just the word (strip probability text)
        parts = strsplit(strtrim(selected));
        word  = parts{1};

        % append selected word to current input
        current = strtrim(inputField.Value);
        if isempty(current)
            inputField.Value = word;
        else
            inputField.Value = [current ' ' word];
        end

        % trigger new search with updated input
        onInputChanged();
    end

    % ---- clear everything ----
    function clearAll()
        inputField.Value = '';
        suggList.Items   = {};
    end

end

% ============================================================
% RUN PART 5
% ============================================================
fprintf('=== PART 5: Application and Documentation ===\n\n');

% --- 1. Save Model ---
fprintf('--- 1. Saving Model ---\n');
saveModel(bigramModel, trigramModel, coMatrix, corpus.vocab);
fprintf('\n');

% --- 2. Load and Verify ---
fprintf('--- 2. Loading Model (verify) ---\n');
[bgModel, tgModel, coMat, voc] = loadModel('nwp_model.mat');
fprintf('Vocab size loaded : %d words\n',  numel(voc));
fprintf('Bigram pairs      : %d\n',        numel(keys(bgModel.bigram)));
fprintf('Trigram pairs     : %d\n',        numel(keys(tgModel.trigram)));
fprintf('\n');

% --- 3. Results Analysis ---
fprintf('--- 3. Results Analysis ---\n');
fprintf('Corpus size    : %d tokens\n',       numel(corpus.tokens));
fprintf('Vocabulary     : %d unique words\n', corpus.vocabSize);
fprintf('Train tokens   : %d  (80%%)\n',      numel(trainTok));
fprintf('Test tokens    : %d  (20%%)\n',      numel(testTok));
fprintf('Bigram pairs   : %d unique pairs\n', numel(keys(bigramModel.bigram)));
fprintf('Trigram pairs  : %d unique pairs\n', numel(keys(trigramModel.trigram)));
fprintf('Vector size    : %dx%d matrix\n',    size(coMatrix,1), size(coMatrix,2));
fprintf('\n');

% --- 4. Suggestions for Improvement ---
fprintf('--- 4. Suggestions for Improvement ---\n\n');
fprintf('1. LARGER CORPUS\n');
fprintf('   Current : %d tokens\n', numel(corpus.tokens));
fprintf('   Suggest : 10,000+ tokens for better accuracy\n\n');
fprintf('2. BETTER SMOOTHING\n');
fprintf('   Current : Laplace add-1\n');
fprintf('   Suggest : Kneser-Ney smoothing\n\n');
fprintf('3. BETTER WORD VECTORS\n');
fprintf('   Current : Co-occurrence matrix\n');
fprintf('   Suggest : Word2Vec or GloVe embeddings\n\n');
fprintf('4. HIGHER N-GRAM\n');
fprintf('   Current : Bigram and Trigram\n');
fprintf('   Suggest : 4-gram or 5-gram for more context\n\n');
fprintf('5. NEURAL NETWORK\n');
fprintf('   Current : Statistical n-gram model\n');
fprintf('   Suggest : LSTM or Transformer\n\n');

% --- 5. Launch UI ---
fprintf('--- 5. Launching UI ---\n');
fprintf('  Bigram  : type 1 word        e.g. "cat"\n');
fprintf('  Trigram : type 2 words       e.g. "the cat"\n');
fprintf('  Vector  : type any words     e.g. "the cat"\n');
fprintf('  Double-click suggestion to append word and search again\n\n');
launchUI(bigramModel, trigramModel, coMatrix, corpus.vocab);



