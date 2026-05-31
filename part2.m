
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
function prob = getProbability(model, w1, w2, lang)
    isKhmer = @(w) ~isempty(regexp(w, '^[\x{1780}-\x{17FF}។]+$', 'once'));
    if nargin == 4
        if strcmp(lang, 'khmer') && ~isKhmer(w1), prob = 0; return; end
        if strcmp(lang, 'english') && isKhmer(w1), prob = 0; return; end
    end
    if ~isKey(model.bigram, w1)
        prob = 0;
        return;
    end
    inner = model.bigram(w1);
    total = model.unigram(w1);
    if isKey(inner, w2)
        prob = inner(w2) / total;
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
    

    dotMask    = ~strcmp(nextWords, '។') & ~strcmp(nextWords, '.');
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

    dotMask    = ~strcmp(nextWords, '។') & ~strcmp(nextWords, '.');
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

% ★ Split each language separately 80/20
[trainEn, testEn] = splitCorpus(corpusEn.tokens, 0.8);
[trainKh, testKh] = splitCorpus(corpusKh.tokens, 0.8);

% ★ Combine after splitting
trainTok = [trainEn, trainKh];
testTok  = [testEn,  testKh];

bigramModel         = buildBigram(trainTok);
fprintf('Bigram model built.\n');
trigramModel        = buildTrigram(trainTok);
fprintf('Trigram model built.\n\n');

% === Bigram Examples ===
fprintf('=== Bigram Examples ===\n');
fprintf('(input 1 word -> predict next word)\n\n');

fprintf('[ English ]\n');
bigramTests = {'the', 'cat', 'a', 'dog', 'bird'};
for i = 1:numel(bigramTests)
    w = bigramTests{i};
    if isKey(bigramModel.bigram, w)
        inner   = bigramModel.bigram(w);
        ws      = keys(inner);
        counts  = cell2mat(values(inner));
        dotMask = ~strcmp(ws, '.') & ~strcmp(ws, '។');
        ws      = ws(dotMask);
        counts  = counts(dotMask);
        if ~isempty(ws)
            total    = sum(counts);
            [~, idx] = max(counts);
            prob     = counts(idx) / total;
            fprintf('"%s" -> "%s"  (prob: %.4f)\n', w, ws{idx}, prob);
        end
    end
end

fprintf('\n[ Khmer ]\n');
bigramTestsKh = {'ភាសា', 'ខ្មែរ', 'ជា', 'នៅ', 'ការ'};
for i = 1:numel(bigramTestsKh)
    w = bigramTestsKh{i};
    if isKey(bigramModel.bigram, w)
        inner   = bigramModel.bigram(w);
        ws      = keys(inner);
        counts  = cell2mat(values(inner));
        dotMask = ~strcmp(ws, '។') & ~strcmp(ws, '.');
        ws      = ws(dotMask);
        counts  = counts(dotMask);
        if ~isempty(ws)
            total    = sum(counts);
            [~, idx] = max(counts);
            prob     = counts(idx) / total;
            fprintf('"%s" -> "%s"  (prob: %.4f)\n', w, ws{idx}, prob);
        end
    else
        fprintf('"%s" -> unknown\n', w);
    end
end
fprintf('\n');

% === Trigram Examples ===
fprintf('=== Trigram Examples ===\n');
fprintf('(input 2 words -> predict next word)\n\n');

fprintf('[ English ]\n');
trigramTests = {{'the','cat'}, {'the','dog'}, {'a','bird'}, {'cat','ate'}, {'the','river'}};
for i = 1:numel(trigramTests)
    w1  = trigramTests{i}{1};
    w2  = trigramTests{i}{2};
    key = [w1 ' ' w2];
    if isKey(trigramModel.trigram, key)
        inner   = trigramModel.trigram(key);
        ws      = keys(inner);
        counts  = cell2mat(values(inner));
        dotMask = ~strcmp(ws, '.') & ~strcmp(ws, '។');
        ws      = ws(dotMask);
        counts  = counts(dotMask);
        if ~isempty(ws)
            [~, idx] = max(counts);
            total    = sum(counts);
            prob     = counts(idx) / total;
            fprintf('"%s %s" -> "%s"  (prob: %.4f)\n', w1, w2, ws{idx}, prob);
        end
    else
        fprintf('"%s %s" -> unknown\n', w1, w2);
    end
end

fprintf('\n[ Khmer ]\n');
trigramTestsKh = {{'ភាសា','ខ្មែរ'}, {'ជា','ភាសា'}, {'នៅ','ក្នុង'}, {'ភាសា','ខ្មែរ'}, {'របស់','ជន'}};
for i = 1:numel(trigramTestsKh)
    w1  = trigramTestsKh{i}{1};
    w2  = trigramTestsKh{i}{2};
    key = [w1 ' ' w2];
    if isKey(trigramModel.trigram, key)
        inner   = trigramModel.trigram(key);
        ws      = keys(inner);
        counts  = cell2mat(values(inner));
        dotMask = ~strcmp(ws, '។') & ~strcmp(ws, '.');
        ws      = ws(dotMask);
        counts  = counts(dotMask);
        if ~isempty(ws)
            [~, idx] = max(counts);
            total    = sum(counts);
            prob     = counts(idx) / total;
            fprintf('"%s %s" -> "%s"  (prob: %.4f)\n', w1, w2, ws{idx}, prob);
        end
    else
        fprintf('"%s %s" -> unknown\n', w1, w2);
    end
end
fprintf('\n');


% === Smoothing Examples ===
fprintf('=== Laplace Smoothing ===\n');
fprintf('(handles unseen word pairs)\n\n');

fprintf('[ English ]\n');
testPairsEn = {{'the','cat'}, {'the','dog'}, {'a','man'}};
for i = 1:numel(testPairsEn)
    w1   = testPairsEn{i}{1};
    w2   = testPairsEn{i}{2};
    prob = getLaplaceProb(bigramModel, w1, w2, corpusEn.vocabSize);
    fprintf('Smoothed P("%s"|"%s") = %.4f\n', w2, w1, prob);
end

fprintf('\n[ Khmer ]\n');
testPairsKh = {{'ភាសា','ខ្មែរ'}, {'ជា','ភាសា'}, {'នៅ','ក្នុង'}};
for i = 1:numel(testPairsKh)
    w1   = testPairsKh{i}{1};
    w2   = testPairsKh{i}{2};
    prob = getLaplaceProb(bigramModel, w1, w2, corpusKh.vocabSize);
    fprintf('Smoothed P("%s"|"%s") = %.4f\n', w2, w1, prob);
end
fprintf('\n');