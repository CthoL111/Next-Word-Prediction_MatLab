

fid = fopen('corpus_khmer.txt', 'r', 'n', 'UTF-8');
rawKh = fread(fid, '*char')';


% check first 100 characters
fprintf('%s\n', rawKh(1:min(100000, numel(rawKh))));