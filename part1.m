% PART 1: Data Preparation
% Converts raw text into tokens, vocabulary, and frequency map

function corpus = loadAndPreprocess(text, lang)
    if strcmp(lang, 'khmer')
        text = regexprep(text, '[^\x{1780}-\x{17FF}\s។]', '');
        text = regexprep(text, '។', ' ។ ');
    else
        text = lower(text);
        text = regexprep(text, '[^a-z\s\.]', '');
        text = regexprep(text, '\.', ' . ');
    end
    tokens = strsplit(strtrim(text));
    tokens = tokens(~cellfun('isempty', tokens));

    corpus.tokens    = tokens;
    corpus.vocab     = unique(tokens);
    corpus.vocabSize = numel(corpus.vocab);
    corpus.lang      = lang;

    corpus.freqMap = containers.Map();
    for i = 1:numel(tokens)
        w = tokens{i};
        if isKey(corpus.freqMap, w)
            corpus.freqMap(w) = corpus.freqMap(w) + 1;
        else
            corpus.freqMap(w) = 1;
        end
    end

    fprintf('Vocabulary size : %d\n', corpus.vocabSize);
    fprintf('Total tokens    : %d\n', numel(tokens));
end

% ── Load English corpus ──────────────────────────────────────
rawEn = fileread('corpus.txt');
corpusEn = loadAndPreprocess(rawEn, 'english');

% ── Load Khmer corpus ────────────────────────────────────────
fid = fopen('corpus_khmer.txt', 'r', 'n', 'UTF-8');
rawKh = fread(fid, '*char')';
fclose(fid);
corpusKh = loadAndPreprocess(rawKh, 'khmer');

% ── Merge both into one corpus ───────────────────────────────
corpus.tokens    = [corpusEn.tokens, corpusKh.tokens];
corpus.vocab     = unique(corpus.tokens);
corpus.vocabSize = numel(corpus.vocab);
corpus.lang      = 'both';

% Rebuild combined freqMap
corpus.freqMap = containers.Map();
for i = 1:numel(corpus.tokens)
    w = corpus.tokens{i};
    if isKey(corpus.freqMap, w)
        corpus.freqMap(w) = corpus.freqMap(w) + 1;
    else
        corpus.freqMap(w) = 1;
    end
end

fprintf('\n=== COMBINED CORPUS ===\n');
fprintf('Vocabulary size : %d\n', corpus.vocabSize);
fprintf('Total tokens    : %d\n', numel(corpus.tokens));

% ── Top word frequencies ─────────────────────────────────────
words  = keys(corpus.freqMap);
counts = cell2mat(values(corpus.freqMap));
[sortedCounts, idx] = sort(counts, 'descend');
sortedWords = words(idx);

fprintf('Most common words: ');
fprintf('%s ', sortedWords{1:min(5, numel(sortedWords))});
fprintf('\n\n');