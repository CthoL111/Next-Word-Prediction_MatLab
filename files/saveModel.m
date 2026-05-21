% PART 5: Save trained model to a .mat file
function saveModel(bigramModel, trigramModel, coMatrix, vocab, filename)
    if nargin < 5, filename = 'nwp_model.mat'; end
    save(filename, 'bigramModel', 'trigramModel', 'coMatrix', 'vocab');
    fprintf('Model saved to  : %s\n', filename);
end
