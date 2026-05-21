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
    corpus.posMap = tagPartsOfSpeech(tokens);
    fprintf('Vocabulary size : %d\n', corpus.vocabSize);
    fprintf('Total tokens    : %d\n', numel(tokens));
end

function posMap = tagPartsOfSpeech(tokens)
    articles     = {'a','an','the'};
    prepositions = {'in','on','at','by','to','of','from',...
                    'over','under','near','with','into'};
    conjunctions = {'and','but','or','so','because','when',...
                    'if','although','while','that'};
    pronouns     = {'i','he','she','they','we','it',...
                    'him','her','them','us','his','its'};
    beVerbs      = {'is','are','was','were','am','be','been'};
    verbs        = {'ate','saw','ran','flew','swam','chased',...
                    'jumped','played','caught','barked','sat',...
                    'watched','walked','said','went','came',...
                    'looked','found','made','think','know'};
    nouns        = {'cat','dog','bird','fish','river','mat',...
                    'floor','day','man','woman','house','tree',...
                    'water','food','time','people','world'};
    adjectives   = {'big','small','fast','slow','deep','wide',...
                    'clean','high','good','bad','old','new'};
    adverbs      = {'quickly','silently','away','together',...
                    'high','well','fast','back'};

    posMap = containers.Map();
    for i = 1:numel(tokens)
        w = tokens{i};
        if ismember(w, articles)
            tag = 'ART';
        elseif ismember(w, prepositions)
            tag = 'PREP';
        elseif ismember(w, conjunctions)
            tag = 'CONJ';
        elseif ismember(w, pronouns)
            tag = 'PRON';
        elseif ismember(w, beVerbs)
            tag = 'BVERB';
        elseif ismember(w, verbs)
            tag = 'VERB';
        elseif ismember(w, nouns)
            tag = 'NOUN';
        elseif ismember(w, adjectives)
            tag = 'ADJ';
        elseif ismember(w, adverbs)
            tag = 'ADV';
        else
            tag = 'UNK';
        end
        posMap(w) = tag;
    end
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