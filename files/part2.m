if ~exist('corpus', 'var') || ~isfield(corpus, 'posMap')
    fprintf('Running Part 1 first...\n');
    run('part1.m');  % ← change to your actual Part 1 filename
end
% splitCorpus
function [trainTok, testTok] = splitCorpus(tokens, ratio)
    totalWords = numel(tokens);
    cutPoint = floor(totalWords * ratio);
    trainTok = tokens(1:cutPoint);
    testTok  = tokens(cutPoint+1:end);
end

% buildBigram 
function model = buildBigram(tokens)
    model.bigram  = containers.Map();
    model.unigram = containers.Map();

    for i = 1:numel(tokens)-1
        w1 = tokens{i};
        w2 = tokens{i+1};

        if isKey(model.unigram, w1)
            model.unigram(w1) = model.unigram(w1) + 1;
        else
            model.unigram(w1) = 1;
        end

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

% buildTrigram (
function model = buildTrigram(tokens)
    model.trigram = containers.Map();
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

% getProbability 
function prob = getProbability(model, w1, w2)
    if ~isKey(model.bigram, w1)
        prob = 0;
        return;
    end
    inner = model.bigram(w1);
    total = model.unigram(w1);
    if isKey(inner, w2)
        prob = inner(w2) / total;  % count / total = probability
    else
        prob = 0;
    end
end

% predictBigram 
function nextWord = predictBigram(model, word)
    if ~isKey(model.bigram, word)
        nextWord = 'unknown';
        return;
    end
    inner      = model.bigram(word);
    nextWords  = keys(inner);
    nextCounts = cell2mat(values(inner));
    [~, idx]   = max(nextCounts);
    nextWord   = nextWords{idx};
end

% predictTrigram 
function nextWord = predictTrigram(model, word1, word2)
    key = [word1 ' ' word2];
    if ~isKey(model.trigram, key)
        nextWord = 'unknown';
        return;
    end
    inner      = model.trigram(key);
    nextWords  = keys(inner);
    nextCounts = cell2mat(values(inner));
    [~, idx]   = max(nextCounts);
    nextWord   = nextWords{idx};
end

% getLaplaceProb 
function prob = getLaplaceProb(model, w1, w2, vocabSize)
    if isKey(model.bigram, w1)
        inner = model.bigram(w1);
        if isKey(inner, w2)
            count = inner(w2);  % seen before
        else
            count = 0;          % never seen
        end
        total = model.unigram(w1);
    else
        count = 0;
        total = 0;
    end
    % +1 count +vocabSize total → never zero!
    prob = (count + 1) / (total + vocabSize);
end

% buildGrammarRules
function rules = buildGrammarRules()
    rules = containers.Map();
    rules('ART')   = {'ADJ','NOUN'};
    rules('NOUN')  = {'VERB','PREP','CONJ','BVERB'};
    rules('VERB')  = {'NOUN','ART','PREP','ADV'};
    rules('ADJ')   = {'NOUN'};
    rules('PREP')  = {'ART','NOUN'};
    rules('PRON')  = {'VERB','BVERB'};
    rules('BVERB') = {'ADJ','NOUN','ADV'};
    rules('ADV')   = {'VERB','ADJ'};
    rules('CONJ')  = {'NOUN','ART','PRON'};
end

% predictGrammar
function nextWord = predictGrammar(bigramModel, posMap, word, topK, silent)
    if nargin < 4, topK   = 3;     end
    if nargin < 5, silent = false; end

    % check word exists
    if ~isKey(bigramModel.bigram, word)
        nextWord = {};
        if ~silent
            fprintf('Word "%s" not found.\n', word);
        end
        return;
    end

    % get grammar tag of input word
    if isKey(posMap, word)
        currentTag = posMap(word);
    else
        currentTag = 'UNK';
    end

    % get allowed next tags
    rules = buildGrammarRules();
    if isKey(rules, currentTag)
        allowedTags = rules(currentTag);
    else
        allowedTags = {};
    end

    % get all candidates from bigram
    inner      = bigramModel.bigram(word);
    ws         = keys(inner);
    counts     = cell2mat(values(inner));

    % split into grammar correct and incorrect
    grammarOK     = {};
    grammarCnt    = [];
    grammarBad    = {};

    for i = 1:numel(ws)
        candidate = ws{i};
        if isKey(posMap, candidate)
            candTag = posMap(candidate);
        else
            candTag = 'UNK';
        end

        if isempty(allowedTags) || ismember(candTag, allowedTags)
            grammarOK  = [grammarOK,  candidate]; %#ok
            grammarCnt = [grammarCnt, counts(i)]; %#ok
        else
            grammarBad = [grammarBad, candidate]; %#ok
        end
    end

    % sort by count
    [~, idx]  = sort(grammarCnt, 'descend');
    grammarOK = grammarOK(idx);
    topK      = min(topK, numel(grammarOK));
    nextWord  = grammarOK(1:topK);

    % print if not silent
    if ~silent
        fprintf('Grammar prediction after "%s" [%s]:\n', word, currentTag);
        fprintf('Allowed tags: ');
        fprintf('%s ', allowedTags{:});
        fprintf('\n\n');
        fprintf('CORRECT:\n');
        for k = 1:numel(grammarOK)
            tag = 'UNK';
            if isKey(posMap, grammarOK{k})
                tag = posMap(grammarOK{k});
            end
            fprintf('  %d. %-12s [%s]\n', k, grammarOK{k}, tag);
        end
        if ~isempty(grammarBad)
            fprintf('FILTERED OUT:\n');
            for k = 1:numel(grammarBad)
                tag = 'UNK';
                if isKey(posMap, grammarBad{k})
                    tag = posMap(grammarBad{k});
                end
                fprintf('  x. %-12s [%s]\n', grammarBad{k}, tag);
            end
        end
    end
end

% build models 
fprintf('=== PART 2: N-gram Models ===\n');
[trainTok, testTok] = splitCorpus(corpus.tokens, 0.8);
bigramModel         = buildBigram(trainTok);
fprintf('Bigram model built.\n');
trigramModel        = buildTrigram(trainTok);
fprintf('Trigram model built.\n\n');

% show probability examples
fprintf('=== Transition Probabilities ===\n');
testPairs = {{'the','cat'}, {'the','dog'}, {'a','man'}};
for i = 1:numel(testPairs)
    w1   = testPairs{i}{1};
    w2   = testPairs{i}{2};
    prob = getProbability(bigramModel, w1, w2);
    fprintf('P("%s"|"%s") = %.4f\n', w2, w1, prob);
end
fprintf('\n');

% show smoothing examples
fprintf('=== Laplace Smoothing ===\n');
vocabSize = corpus.vocabSize;
for i = 1:numel(testPairs)
    w1   = testPairs{i}{1};
    w2   = testPairs{i}{2};
    prob = getLaplaceProb(bigramModel, w1, w2, vocabSize);
    fprintf('Smoothed P("%s"|"%s") = %.4f\n', w2, w1, prob);
end
fprintf('\n');

% GRAMMAR TEST
fprintf('=== Grammar-Aware Predictions ===\n\n');
testWords = {'the', 'cat', 'a', 'bird'};
for i = 1:numel(testWords)
    predictGrammar(bigramModel, corpus.posMap, testWords{i}, 3);
    fprintf('\n');
end

fprintf('=== INTERACTIVE NEXT WORD PREDICTOR ===\n');
fprintf('1 word input  → Bigram prediction\n');
fprintf('2 word input  → Trigram prediction\n');
fprintf('Type "quit"   → Stop\n\n');

while true
    userInput  = input('Enter word(s): ', 's');

    if strcmp(userInput, 'quit')
        fprintf('Goodbye!\n');
        break;
    end

    inputWords = strsplit(strtrim(userInput));
    numWords   = numel(inputWords);

    if numWords == 1
        word   = inputWords{1};

        % normal bigram
        result = predictBigram(bigramModel, word);
        prob   = getProbability(bigramModel, word, result);
        fprintf('Bigram  : "%s" → "%s" (prob: %.4f)\n', word, result, prob);

        % ↓↓↓ ADD GRAMMAR HERE ↓↓↓
        gramPreds = predictGrammar(bigramModel, corpus.posMap, word, 3, true);
        if ~isempty(gramPreds)
            fprintf('Grammar : "%s" → ', word);
            fprintf('"%s" ', gramPreds{:});
            fprintf('\n');
        else
            fprintf('Grammar : no grammar predictions for "%s"\n', word);
        end
        % ↑↑↑ ADD GRAMMAR HERE ↑↑↑

        fprintf('\n');

    elseif numWords >= 2
        word1  = inputWords{numWords-1};
        word2  = inputWords{numWords};
        result = predictTrigram(trigramModel, word1, word2);
        fprintf('Trigram : "%s %s" → "%s"\n\n', word1, word2, result);
    end
end