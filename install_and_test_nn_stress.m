
function stsv = stress_nn(stnv, ym, pr, stnt, stnb)
%==================================================================================================
%  Neural Network replacement for the Complete AVIB model constitutive law
%  This function uses a pre-trained PyTorch model to predict stress from strain
%
%  Input:
%     stnv(3) = strain vector [stnv_xx, stnv_yy, stnv_xy]
%          ym = Young's modulus
%          pr = Poisson ratio
%        stnt = tensile strain strength
%        stnb = compressive strain strength
%  Output:
%     stsv(3) = stress vector [stsv_xx, stsv_yy, stsv_xy]
%==================================================================================================
    persistent model X_scaler y_scaler is_initialized;
    
    % Initialize the model and scalers on first call
    if isempty(is_initialized) || ~is_initialized
        % Paths to model files
        model_dir = 'stress_model';
        model_path = fullfile(model_dir, 'stress_model.pt');
        X_scaler_path = fullfile(model_dir, 'X_scaler.pkl');
        y_scaler_path = fullfile(model_dir, 'y_scaler.pkl');
        
        try
            % Load the PyTorch model using py.torch.jit.load
            if ~exist('py', 'builtin')
                error('Python interface is not available. Please install Python support for MATLAB.');
            end
            
            % Import necessary Python modules
            py.importlib.import_module('torch');
            py.importlib.import_module('joblib');
            py.importlib.import_module('numpy');
            
            % Load the model and scalers
            model = py.torch.jit.load(model_path);
            X_scaler = py.joblib.load(X_scaler_path);
            y_scaler = py.joblib.load(y_scaler_path);
            
            is_initialized = true;
            fprintf('Neural network model loaded successfully.\n');
        catch e
            error('Error loading neural network model: %s\n', e.message);
        end
    end
    
    % Prepare input data as a row vector
    input_data = [stnv(1), stnv(2), stnv(3), ym, pr, stnt, stnb];
    
    % Convert to Python numpy array
    input_np = py.numpy.array(input_data);
    
    % Reshape to 2D array (required for sklearn transformers)
    input_reshaped = input_np.reshape(int32(1), int32(-1));
    
    % Scale the inputs
    input_scaled = X_scaler.transform(input_reshaped);
    
    % Convert to PyTorch tensor
    input_tensor = py.torch.tensor(input_scaled, pyargs('dtype', py.torch.float32));
    
    % Run inference
    model.eval();
    with_no_grad = py.torch.no_grad();
    with_no_grad.__enter__();
    try
        output_tensor = model.forward(input_tensor);
        % Convert back to numpy array
        output_np = output_tensor.cpu().numpy();
    catch e
        with_no_grad.__exit__(e, py.None, py.None);
        rethrow(e);
    end
    with_no_grad.__exit__(py.None, py.None, py.None);
    
    % Inverse transform to get original scale
    output_scaled = y_scaler.inverse_transform(output_np);
    
    % Convert to MATLAB array
    output_matlab = double(py.array.array('d', py.numpy.nditer(output_scaled)));
    
    % Return result in the same format as the original stress function
    stsv = reshape(output_matlab, 3, 1);
end

function stsv = stress_fallback(stnv, ym, pr, stnt, stnb)
%==================================================================================================
% Fallback implementation of the original stress function in case the NN model fails
%==================================================================================================
    % Gauss points for integration
    gx = [-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142];
    gh = [0.1713244924, 0.3607615730, 0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924];
    cf = [0.39269908, 1.178097245, 1.963495085, 2.74889357]; % pi/8, 3pi/8, 5pi/8, 7pi/8

    % Initialize
    d = zeros(3, 3);
    stsv = zeros(3, 1);

    % nonlinear model
    lama = ym/(pi*(1.0-pr));
    lamb = ym*(1.0-3.0*pr)/(pi*(1.0-pr)*(1.0+pr));
    q = (1.0-3.0*pr)/(2.0*(1.0+pr));

    c = zeros(4, 1);
    for j1 = 1:4
        for i1 = 1:6
            thi = 0.39269908*gx(i1)+cf(j1);
            x1 = sin(thi); x2 = cos(thi); % micro bond direction vector x=(x1,x2)
            y1 = stnv(1)*x1+0.5*stnv(3)*x2; % y=estn*x (y1,y2)
            y2 = 0.5*stnv(3)*x1+stnv(2)*x2;
            dn = x1*y1+x2*y2; % bond stretch
            rt = y1*y1+y2*y2-dn*dn; % bond rotation^2
            stc = stnt; % tension
            if dn < 0.0
                stc = -stnb; % compression
            end
            c1 = exp(-dn/stc);
            c2 = exp(-rt/(stc*stc));
            fa = lama*dn*c1*(1.0-q+q*c2);
            fb = lamb*c1*(1.0+dn/stc)*c2;
            a = zeros(4, 1);
            a(1) = fa*x1*x1 + fb*(y1*x1 - dn*x1*x1); % s11
            a(2) = fa*x1*x2 + fb*(y1*x2 - dn*x1*x2); % s12
            a(3) = fa*x2*x1 + fb*(y2*x1 - dn*x2*x1); % s21
            a(4) = fa*x2*x2 + fb*(y2*x2 - dn*x2*x2); % s22
            c = c + a*0.39269908*gh(i1);
        end
    end

    stsv(1) = 2.0*c(1); % 2*pi
    stsv(2) = 2.0*c(4);
    stsv(3) = c(2)+c(3);
end

function test_stress_nn_performance()
% Test function to compare performance between original and NN stress functions
    
    fprintf('Testing performance of neural network stress function vs original implementation\n');
    
    % Set up test parameters
    n_tests = 1000;
    
    % Generate random test inputs
    rng(42); % For reproducibility
    stnv_list = 0.05 * (rand(3, n_tests) - 0.5);
    ym_list = 1e9 * (1 + 99 * rand(1, n_tests)); % 1e9 to 100e9
    pr_list = 0.1 + 0.35 * rand(1, n_tests);     % 0.1 to 0.45
    stnt_list = 0.001 + 0.099 * rand(1, n_tests); % 0.001 to 0.1
    stnb_list = 0.001 + 0.099 * rand(1, n_tests); % 0.001 to 0.1
    
    % Preallocate results
    original_results = zeros(3, n_tests);
    nn_results = zeros(3, n_tests);
    
    % Warm up
    stnv = stnv_list(:, 1);
    ym = ym_list(1);
    pr = pr_list(1);
    stnt = stnt_list(1);
    stnb = stnb_list(1);
    stress_fallback(stnv, ym, pr, stnt, stnb);
    stress_nn(stnv, ym, pr, stnt, stnb);
    
    % Test original function
    fprintf('Testing original stress function...\n');
    tic;
    for i = 1:n_tests
        stnv = stnv_list(:, i);
        ym = ym_list(i);
        pr = pr_list(i);
        stnt = stnt_list(i);
        stnb = stnb_list(i);
        
        original_results(:, i) = stress_fallback(stnv, ym, pr, stnt, stnb);
    end
    original_time = toc;
    fprintf('Original function: %f seconds for %d evaluations (%.6f sec/eval)\n', ...
            original_time, n_tests, original_time/n_tests);
    
    % Test neural network function
    fprintf('Testing neural network stress function...\n');
    tic;
    for i = 1:n_tests
        stnv = stnv_list(:, i);
        ym = ym_list(i);
        pr = pr_list(i);
        stnt = stnt_list(i);
        stnb = stnb_list(i);
        
        nn_results(:, i) = stress_nn(stnv, ym, pr, stnt, stnb);
    end
    nn_time = toc;
    fprintf('Neural network function: %f seconds for %d evaluations (%.6f sec/eval)\n', ...
            nn_time, n_tests, nn_time/n_tests);
    
    % Calculate speedup
    speedup = original_time / nn_time;
    fprintf('Speedup factor: %.2fx\n', speedup);
    
    % Calculate error statistics
    abs_diff = abs(original_results - nn_results);
    rel_diff = abs_diff ./ (abs(original_results) + 1e-10);
    
    mean_abs_error = mean(abs_diff, 'all');
    max_abs_error = max(abs_diff, [], 'all');
    mean_rel_error = mean(rel_diff, 'all') * 100;
    max_rel_error = max(rel_diff, [], 'all') * 100;
    
    fprintf('\nError Statistics:\n');
    fprintf('Mean Absolute Error: %e\n', mean_abs_error);
    fprintf('Max Absolute Error: %e\n', max_abs_error);
    fprintf('Mean Relative Error: %.4f%%\n', mean_rel_error);
    fprintf('Max Relative Error: %.4f%%\n', max_rel_error);
    
    % Create error plots
    figure;
    subplot(3,1,1);
    plot(1:n_tests, original_results(1,:), 'b-', 1:n_tests, nn_results(1,:), 'r--');
    title('Stress XX Component');
    legend('Original', 'Neural Network');
    grid on;
    
    subplot(3,1,2);
    plot(1:n_tests, original_results(2,:), 'b-', 1:n_tests, nn_results(2,:), 'r--');
    title('Stress YY Component');
    grid on;
    
    subplot(3,1,3);
    plot(1:n_tests, original_results(3,:), 'b-', 1:n_tests, nn_results(3,:), 'r--');
    title('Stress XY Component');
    grid on;
    
    % Save the figure
    saveas(gcf, 'stress_model/stress_comparison.png');
    
    % Create a scatter plot of predicted vs actual
    figure;
    subplot(1,3,1);
    scatter(original_results(1,:), nn_results(1,:), '.');
    title('Stress XX: Predicted vs Actual');
    xlabel('Original'); ylabel('Neural Network');
    grid on; axis equal;
    refline(1,0);
    
    subplot(1,3,2);
    scatter(original_results(2,:), nn_results(2,:), '.');
    title('Stress YY: Predicted vs Actual');
    xlabel('Original'); ylabel('Neural Network');
    grid on; axis equal;
    refline(1,0);
    
    subplot(1,3,3);
    scatter(original_results(3,:), nn_results(3,:), '.');
    title('Stress XY: Predicted vs Actual');
    xlabel('Original'); ylabel('Neural Network');
    grid on; axis equal;
    refline(1,0);
    
    % Save the figure
    saveas(gcf, 'stress_model/stress_scatter.png');
end

function install_and_test_nn_stress()
% Helper function to install required Python packages and test the neural network stress function
    try
        % Check if Python is installed and accessible
        if ~exist('py', 'builtin')
            error('Python interface is not available. Please install Python support for MATLAB.');
        end
        
        % Check if PyTorch is installed
        try
            py.importlib.import_module('torch');
            fprintf('PyTorch is already installed.\n');
        catch
            fprintf('PyTorch not found. Installing...\n');
            system('pip install torch torchvision');
        end
        
        % Check if joblib is installed
        try
            py.importlib.import_module('joblib');
            fprintf('joblib is already installed.\n');
        catch
            fprintf('joblib not found. Installing...\n');
            system('pip install joblib');
        end
        
        % Check if numpy is installed
        try
            py.importlib.import_module('numpy');
            fprintf('NumPy is already installed.\n');
        catch
            fprintf('NumPy not found. Installing...\n');
            system('pip install numpy');
        end
        
        % Check if scikit-learn is installed
        try
            py.importlib.import_module('sklearn');
            fprintf('scikit-learn is already installed.\n');
        catch
            fprintf('scikit-learn not found. Installing...\n');
            system('pip install scikit-learn');
        end
        
        % Test the neural network stress function
        fprintf('\nTesting neural network stress function...\n');
        stnv = [0.01; -0.005; 0.003];
        ym = 1e9;
        pr = 0.3;
        stnt = 0.01;
        stnb = 0.01;
        
        % Run both implementations
        original_result = stress_fallback(stnv, ym, pr, stnt, stnb);
        nn_result = stress_nn(stnv, ym, pr, stnt, stnb);
        
        % Compare results
        fprintf('Original stress function result:\n');
        disp(original_result);
        fprintf('Neural network stress function result:\n');
        disp(nn_result);
        
        % Calculate relative error
        abs_diff = abs(original_result - nn_result);
        rel_diff = abs_diff ./ (abs(original_result) + 1e-10) * 100;
        
        fprintf('Absolute difference: %e, %e, %e\n', abs_diff(1), abs_diff(2), abs_diff(3));
        fprintf('Relative difference: %.4f%%, %.4f%%, %.4f%%\n', rel_diff(1), rel_diff(2), rel_diff(3));
        
        % Run performance test
        test_stress_nn_performance();
        
        fprintf('\nNeural network stress function installation and testing completed successfully!\n');
    catch e
        fprintf('Error: %s\n', e.message);
    end
end
