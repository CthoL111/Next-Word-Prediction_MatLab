% PART 1: Data Preparation
% Converts raw text into tokens, vocabulary, and frequency map
function corpus = loadAndPreprocess(text)
    text   = lower(text);                          % Step 1: lowercase
    text   = regexprep(text, '[^a-z\s]', '');      % Step 2: remove punctuation
    tokens = strsplit(strtrim(text));              % Step 3: split into words
    tokens = tokens(~cellfun('isempty', tokens));  % Step 4: remove empty

    corpus.tokens    = tokens;
    corpus.vocab     = unique(tokens);             % unique word list
    corpus.vocabSize = numel(corpus.vocab);

    % Count frequency of each word
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