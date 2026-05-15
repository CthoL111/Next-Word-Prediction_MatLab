
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
        result = predictBigram(bigramModel, word);
        prob   = getProbability(bigramModel, word, result);
        fprintf('Bigram  : "%s" → "%s" (prob: %.4f)\n\n', word, result, prob);

    elseif numWords >= 2
        word1  = inputWords{numWords-1};
        word2  = inputWords{numWords};
        result = predictTrigram(trigramModel, word1, word2);
        fprintf('Trigram : "%s %s" → "%s"\n\n', word1, word2, result);
    end
end