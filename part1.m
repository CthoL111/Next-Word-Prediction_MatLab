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
    if strcmp(lang, 'khmer')
        fprintf('\n=== Khmer CORPUS ===\n');
    else
        fprintf('\n=== English CORPUS ===\n');
    end
    fprintf('Vocabulary size : %d\n', corpus.vocabSize);
    fprintf('Total tokens    : %d\n', numel(tokens));
    % top 5 words per language
    w_keys = keys(corpus.freqMap);
    w_vals = cell2mat(values(corpus.freqMap));
    [~, widx] = sort(w_vals, 'descend');
    fprintf('Most common     : ');
    fprintf('%s ', w_keys{widx(1:min(5,numel(w_keys)))});
    fprintf('\n');
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
fprintf('All vocabulary size : %d\n', corpus.vocabSize);
fprintf('All tokens    : %d\n', numel(corpus.tokens));
fprintf('\n\n');

% ── Build sorted Khmer vocab for segmentation (longest match first) ──
khmerVocab = corpus.vocab(cellfun(@(w) ...
    ~isempty(regexp(w, '^[\x{1780}-\x{17FF}។]+$', 'once')), corpus.vocab));
[~, lenIdx] = sort(cellfun(@numel, khmerVocab), 'descend');
khmerVocab  = khmerVocab(lenIdx);