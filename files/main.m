
% ============================================================
% NEXT WORD PREDICTION - MAIN SCRIPT
% Run this file to execute the full project
% ============================================================
clc; clear; close all;

%% ---- PART 1: Data Preparation (15%) ----
fprintf('=== PART 1: Data Preparation ===\n');
rawText = fileread('corpus.txt');
corpus  = loadAndPreprocess(rawText);

% Visualize top word frequencies
words  = keys(corpus.freqMap);
counts = cell2mat(values(corpus.freqMap));
[sortedCounts, idx] = sort(counts, 'descend');
sortedWords = words(idx);
topN = min(20, numel(sortedWords));

figure('Name', 'Word Frequency Distribution');
bar(sortedCounts(1:topN));
set(gca, 'XTick', 1:topN, 'XTickLabel', sortedWords(1:topN), 'XTickLabelRotation', 45);
title('Top Word Frequencies');
xlabel('Word'); ylabel('Count'); grid on;

fprintf('Most common words: ');
fprintf('%s ', sortedWords{1:min(5,topN)});
fprintf('\n\n');

%% ---- PART 2: N-gram Model (30%) ----
fprintf('=== PART 2: N-gram Models ===\n');
[trainTok, testTok] = splitCorpus(corpus.tokens, 0.8);

bigramModel  = buildBigram(trainTok);
fprintf('Bigram model built.\n');

trigramModel = buildTrigram(trainTok);
fprintf('Trigram model built.\n\n');

%% ---- PART 3: Vector Representation (25%) ----
fprintf('=== PART 3: Vector Representations ===\n');
coMatrix = buildCoOccurrence(trainTok, corpus.vocab, 2);
fprintf('Co-occurrence matrix built: %dx%d\n\n', size(coMatrix,1), size(coMatrix,2));

%% ---- PART 4: Model Evaluation (15%) ----
fprintf('=== PART 4: Evaluation ===\n');
evaluateBigram(bigramModel, testTok);
perplexity(bigramModel, testTok);
fprintf('\n');

%% ---- PART 5: Application (15%) ----
fprintf('=== PART 5: Saving Model & Launching UI ===\n');
saveModel(bigramModel, trigramModel, coMatrix, corpus.vocab);
fprintf('Opening UI...\n');
launchUI(bigramModel, trigramModel, coMatrix, corpus.vocab);
