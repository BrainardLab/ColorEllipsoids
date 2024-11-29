clear all; close all; clc

%

% Specify output Excel file
outputExcelFile = 'session_orders.xlsx';

%% Generate session order for AEPsych trials 
nSubjs = 5; % Number of participants
idx_AEPsych = [1,3,5,11,13,15,21,23,25]; 
nRefs_AEPsych = length(idx_AEPsych);
idx_MOCS = 1:8; 
nRefs_MOCS = length(idx_MOCS);
% Preallocate structures for output
sessionData = struct;

for n = 1:nSubjs
    % Generate session order for AEPsych
    rng(n); 
    session_order_AEPsych = randperm(nRefs_AEPsych);
    idx_shuffled_AEPsych = idx_AEPsych(session_order_AEPsych);
    
    %% Generate session order for MOCS trials
    rng(n);
    session_order_MOCS = randperm(nRefs_MOCS);
    idx_shuffled_MOCS = idx_MOCS(session_order_MOCS);
    
    % Pad shorter arrays with NaN
    maxLength = max(nRefs_AEPsych, nRefs_MOCS);
    padded_Trial_AEPsych = (1:nRefs_AEPsych)';
    padded_Idx_AEPsych = idx_shuffled_AEPsych';
    padded_Order_AEPsych = session_order_AEPsych';
    
    padded_Trial_MOCS = (1:nRefs_MOCS)';
    padded_Idx_MOCS = idx_shuffled_MOCS';
    padded_Order_MOCS = session_order_MOCS';
    
    if nRefs_AEPsych < maxLength
        padded_Trial_AEPsych = [padded_Trial_AEPsych; NaN(maxLength - nRefs_AEPsych, 1)];
        padded_Idx_AEPsych = [padded_Idx_AEPsych; NaN(maxLength - nRefs_AEPsych, 1)];
        padded_Order_AEPsych = [padded_Order_AEPsych; NaN(maxLength - nRefs_AEPsych, 1)];
    end
    
    if nRefs_MOCS < maxLength
        padded_Trial_MOCS = [padded_Trial_MOCS; NaN(maxLength - nRefs_MOCS, 1)];
        padded_Idx_MOCS = [padded_Idx_MOCS; NaN(maxLength - nRefs_MOCS, 1)];
        padded_Order_MOCS = [padded_Order_MOCS; NaN(maxLength - nRefs_MOCS, 1)];
    end
    
    % Store data in a structure
    sessionData(n).Subject = n;
    sessionData(n).session_order_AEPsych = session_order_AEPsych;
    sessionData(n).idx_shuffled_AEPsych = idx_shuffled_AEPsych;
    sessionData(n).session_order_MOCS = session_order_MOCS;
    sessionData(n).idx_shuffled_MOCS = idx_shuffled_MOCS;
    
    % Prepare data for writing to Excel
    T = table(padded_Trial_AEPsych, padded_Idx_AEPsych, padded_Order_AEPsych, ...
              padded_Trial_MOCS, padded_Idx_MOCS, padded_Order_MOCS, ...
              'VariableNames', {'Trial_AEPsych', 'Idx_AEPsych', ...
                                'Order_AEPsych', 'Trial_MOCS', ...
                                'Idx_MOCS', 'Order_MOCS'});
    
    % Write each subject's data to a separate sheet
    sheetName = sprintf('Subject_%d', n);
    writetable(T, outputExcelFile, 'Sheet', sheetName);
end

disp(['Session order file saved as: ' outputExcelFile]);
