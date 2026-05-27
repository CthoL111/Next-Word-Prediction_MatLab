
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
    
    % Remove '.' from candidates
dotMask    = ~strcmp(nextWords, '.');
nextWords  = nextWords(dotMask);
nextCounts = nextCounts(dotMask);
    
    if isempty(nextWords)
        nextWord = 'unknown';
        return;
    end
    [~, idx]  = max(nextCounts);
    nextWord  = nextWords{idx};
end

% predictTrigram 
function [nextWord, prob] = predictTrigram(model, word1, word2)
    key = [word1 ' ' word2];
    if ~isKey(model.trigram, key)
        nextWord = 'unknown';
        prob = 0;
        return;
    end
    inner      = model.trigram(key);
    nextWords  = keys(inner);
    nextCounts = cell2mat(values(inner));

    dotMask    = ~strcmp(nextWords, '.');
    nextWords  = nextWords(dotMask);
    nextCounts = nextCounts(dotMask);

    if isempty(nextWords)
        nextWord = 'unknown';
        prob = 0;
        return;
    end
    [~, idx]  = max(nextCounts);
    nextWord  = nextWords{idx};
    prob      = nextCounts(idx) / sum(nextCounts);  % add this
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



