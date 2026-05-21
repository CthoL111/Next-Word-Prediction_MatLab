% PART 1: Data Preparation
% Converts raw text into tokens, vocabulary, and frequency map
function corpus = loadAndPreprocess(text)
    text   = lower(text);                         
    text   = regexprep(text, '[^a-z\s]', '');     
    tokens = strsplit(strtrim(text));              
    tokens = tokens(~cellfun('isempty', tokens));  

    corpus.tokens    = tokens;
    corpus.vocab     = unique(tokens);             
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
fprintf('=== PART 1: Data Preparation ===\n');
rawText = fileread('corpus.txt');
corpus  = loadAndPreprocess(rawText);

% Visualize top word frequencies
words  = keys(corpus.freqMap);
counts = cell2mat(values(corpus.freqMap));
[sortedCounts, idx] = sort(counts, 'descend');
sortedWords = words(idx);
topN = min(20, numel(sortedWords));

%figure();
%bar(sortedCounts(1:topN));
%set(gca, 'XTick', 1:topN, 'XTickLabel', sortedWords(1:topN), 'XTickLabelRotation', 45);

fprintf('Most common words: ');
fprintf('%s ', sortedWords{1:min(5,numel(sortedWords))});
fprintf('\n\n');