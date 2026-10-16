function saveResults(results, p_name)
    if (exist([pwd filesep 'results.csv'], 'file') == 0)
        fid = fopen([pwd filesep 'results.csv'], 'w');
        if fid == -1
            error('Research:ResultsOpenFailed', 'Cannot open results.csv for writing.');
        end
        fileCleanup = onCleanup(@() fclose(fid));
        fprintf(fid, '%s,%s,%s,%s,%s,%s,%s, %s\n', ...
            'Data Set','Avg Accuracy with BCS', 'Std. Dev',...
            'Avg Accuracy without BCS', 'Std. Dev',...
            'Original Component Size', 'Component size after BCS', 'AddedDiv'...
            );
    elseif (exist([pwd filesep 'results.csv'], 'file') == 2)
        fid = fopen([pwd filesep 'results.csv'], 'a');
        if fid == -1
            error('Research:ResultsOpenFailed', 'Cannot open results.csv for writing.');
        end
        fileCleanup = onCleanup(@() fclose(fid));
    end
    fprintf(fid, '%s, ', p_name);
    fprintf(fid, '%f,%f,%f,%f,%f,%f, %f\n', ...
        results.avgAccuracy, results.stdDevWithBCS,...
        results.avgAccuracy_without_BCS, results.stdDevWithoutBCS,...
        results.originalSize, results.sizeAfterBCS, results.addedDiv...
        );

end



