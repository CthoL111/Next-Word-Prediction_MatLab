%% ============================================================
%  PART 1: Data Preparation
%  Goal: Turn raw text into a list of words and count each one
%% ============================================================

fprintf('=== PART 1: Data Preparation ===\n');

% --- Load the raw text file ---
rawText = fileread('corpus.txt');

% --- Clean and split the text into words ---
corpus = cleanAndTokenize(rawText);

% --- Count how often each word appears ---
corpus = countWordFrequencies(corpus);

% --- Show a bar chart of the most common words ---
plotTopWords(corpus, 20);


%% ------------------------------------------------------------
%  FUNCTION 1: Clean text and split it into individual words
%% ------------------------------------------------------------
function corpus = cleanAndTokenize(rawText)

    % Convert everything to lowercase so "The" and "the" are the same word
    lowercaseText = lower(rawText);

    % Remove anything that isn't a letter or a space (punctuation, numbers, etc.)
    lettersOnly = regexprep(lowercaseText, '[^a-z\s]', '');

    % Split the text into a list of words using whitespace as the separator
    allWords = strsplit(strtrim(lettersOnly));

    % Remove any accidental empty strings from double spaces
    allWords = allWords(~cellfun('isempty', allWords));

    % Build the unique word list (vocabulary)
    uniqueWords = unique(allWords);

    % Save everything into the corpus structure
    corpus.tokens    = allWords;       % every word in order
    corpus.vocab     = uniqueWords;    % one entry per unique word
    corpus.vocabSize = numel(uniqueWords);

    fprintf('Vocabulary size : %d\n', corpus.vocabSize);
    fprintf('Total tokens    : %d\n', numel(allWords));
end


%% ------------------------------------------------------------
%  FUNCTION 2: Count how many times each word appears
%% ------------------------------------------------------------
function corpus = countWordFrequencies(corpus)

    wordCount = containers.Map();   % acts like a dictionary: word → count

    for i = 1:numel(corpus.tokens)
        currentWord = corpus.tokens{i};

        if isKey(wordCount, currentWord)
            wordCount(currentWord) = wordCount(currentWord) + 1;  % seen before → increment
        else
            wordCount(currentWord) = 1;                           % first time → start at 1
        end
    end

    corpus.freqMap = wordCount;
end


%% ------------------------------------------------------------
%  FUNCTION 3: Plot a bar chart of the most frequent words
%% ------------------------------------------------------------
function plotTopWords(corpus, topN)

    % Pull all words and their counts out of the dictionary
    allWords  = keys(corpus.freqMap);
    allCounts = cell2mat(values(corpus.freqMap));

    % Sort from highest count to lowest
    [sortedCounts, sortOrder] = sort(allCounts, 'descend');
    sortedWords = allWords(sortOrder);

    % Only show as many bars as we actually have words for
    topN = min(topN, numel(sortedWords));

    % Draw the bar chart
    figure('Name', 'Word Frequency Distribution');
    bar(sortedCounts(1:topN));
    set(gca, ...
        'XTick',             1:topN, ...
        'XTickLabel',        sortedWords(1:topN), ...
        'XTickLabelRotation', 45);
    title('Top Word Frequencies');
    xlabel('Word');
    ylabel('Count');
    grid on;

    % Print the top 5 words to the console
    fprintf('Most common words: ');
    fprintf('%s ', sortedWords{1:min(5, topN)});
    fprintf('\n\n');
end