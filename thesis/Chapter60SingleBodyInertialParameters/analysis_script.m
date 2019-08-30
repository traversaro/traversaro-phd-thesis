addpath('../quadfit/')
addpath('../common_scripts')
addpath('../plot2svg')

% Stupid workaround for issue http://stackoverflow.com/questions/19268293/matlab-error-cannot-open-with-static-tls
ones(10,10)*ones(10,10);

%input dataset names
datasets_names = {'dataset01_no_added_mass' ...
                  'dataset02_mass_added_back_left' ...
                  'dataset03_mass_aded_front_left' ...
                  'dataset04_mass_in_middle_28_on_top' ...
                  'dataset05_no_cantilever_mass_added_front_right' ...
                  'dataset06_no_cantilever_mass_added_lower_middle_25' ...
                  'dataset07_no_added_mass' ...
                  'dataset08_no_cantilever_mass_added_lower_back' };
              
n_datasets = length(datasets_names);

              
drop_at_start = [ 0.0 ...
                  0.0 ... 
                  0.0 ...
                  0.0 ...
                  0.0 ...
                  0.0 ...
                  0.0 ...
                  0.0 ];
              
drop_at_end = [ 0.0 ...
                  0.0 ... 
                  0.0 ...
                  0.0 ...
                  0.0 ...
                  0.0 ...
                  0.0 ...
                  0.0 ];
              
%% Sensor original calibration data      
Ws = diag([1/32767 1/32767 1/32767 1/32767 1/32767 1/32767]);

%sensor for right leg on iCubGenova03 : SN039
original_calibration_matrix_leg_hex = ...
{ 'fe8c' '13c'  '3ae3' 'cb'   '4f0'  'c354'; ...
  '2d7' '3cac' 'e1c2' 'fdc6' '23a'  'e241';  ...
  'd4bd' '47'   '78'   '2904' '2996' '15b' ; ...
  '58e'  '1935' 'f2b6' 'd9d2' '1f59' 'f491'; ...
  '2bda' 'fe8e' 'eb37' '1458' '1272' '16fe'; ...
  '297'  '295d' '259e' '188'  'fe13' '275d'};

original_calibration_leg_max_val = ...
[ -1668 1915 2100 37 36 24];

original_calibration_matrix_leg = zeros(6,6);
for row=1:6
    for col=1:6
        original_calibration_matrix_leg(row,col) = ...
            hex_calib_to_double(original_calibration_matrix_leg_hex{row,col}); 
    end
end
original_calibration_matrix_leg = diag(original_calibration_leg_max_val)*original_calibration_matrix_leg*Ws;

%sensor for rigt\ht foot on iCubGenova03 : SN142
original_calibration_matrix_foot_hex = ...
{ '3'    'f7'   '3d9f' 'fe06' 'ff81' 'c11f'; ...
  '68'   '3efe' 'e0e2' 'fff1' '21a'  'e0b5'; ...
  '2919' 'fe73' 'ff62' '29b2' '2ac5' 'ffbd'; ...
  'fed5' '1d29' 'f1a2' 'df93' '23bb' 'f1d7'; ...
  'da0b' '100'  'e96b' '1504' '13fa' '15b7'; ...
  'ffc6' '2ad7' '297f' 'ff4c' '3e'   '28aa'};

original_calibration_foot_max_val = ...
[ -1257 1456 1683 30 31 19];

original_calibration_matrix_foot = zeros(6,6);
for row=1:6
    for col=1:6
        original_calibration_matrix_foot(row,col) = ...
            hex_calib_to_double(original_calibration_matrix_foot_hex{row,col}); 
    end
end

original_calibration_matrix_foot = diag(original_calibration_foot_max_val)*original_calibration_matrix_foot*Ws;

%% Experimental information
cube_mass = 0.49158 + 0.01944;
cantilever_mass = 0.19198;
total_mass = cube_mass + cantilever_mass;
beam_upper_plane_wrt_foot_sensor_z = 0.014;
cantilever_beam_side = 0.03;
cantilever_com_wrt_to_base = 0.049;
cube_side_position_in_cantilever_wrt_to_base = 0.11;
cube_side = 0.04;
dataset03_cantilever_x_position_wrt_beam_begin = 0.24;
dataset04_cantilever_x_position_wrt_beam_begin = 0.28;
triangle_junction_side = 0.027;
back_foot_sensor_offset_x = 0.06;
cube_com_wrt_base = 0.0197; %considering all we do not get 0.02
beam_length = 0.47; %VERIFYYYY!!!!
dataset06_cube_x_position_wrt_beam_begin = 0.25;
ft_leg_position_wrt_ft_foot_x = 0.0035;
ft_leg_position_wrt_ft_foot_z = -0.3636; %%%VVVVVVVVEEERRRIIFFFYYY!!!


added_mass = [ 0.0 ...
               cantilever_mass+cube_mass ...
               cantilever_mass+cube_mass ...
               cantilever_mass+cube_mass ...
               cube_mass ...
               cube_mass ...
               0.0 ...
               cube_mass ];
           
       
added_center_of_mass_wrt_foot{1} = [0.0 0.0 0.0];

added_center_of_mass_wrt_foot{2} = ...
                  [-0.018 ...
                   0.015 + (0.049*cantilever_mass + 0.11*cube_mass)/total_mass ...
                   beam_upper_plane_wrt_foot_sensor_z + cantilever_beam_side/2 - (0.015+cube_com_wrt_base)*cube_mass/total_mass ];
               
added_center_of_mass_wrt_foot{3} = added_center_of_mass_wrt_foot{2};
                 added_center_of_mass_wrt_foot{3}(1) = added_center_of_mass_wrt_foot{3}(1) + dataset03_cantilever_x_position_wrt_beam_begin;

added_center_of_mass_wrt_foot{4} = ...
                  [ -back_foot_sensor_offset_x+dataset04_cantilever_x_position_wrt_beam_begin+triangle_junction_side+cantilever_beam_side/2 ...
                    (cantilever_beam_side/2 + cube_com_wrt_base)*(cube_mass/total_mass) ...
                    beam_upper_plane_wrt_foot_sensor_z-(cantilever_com_wrt_to_base*cantilever_mass+(cube_side_position_in_cantilever_wrt_to_base+cube_side/2)*cube_mass)/total_mass ];
                 
added_center_of_mass_wrt_foot{5} = ...
    [ -back_foot_sensor_offset_x+beam_length-cube_side/2 ...
      -cantilever_beam_side/2-cube_com_wrt_base ...
      beam_upper_plane_wrt_foot_sensor_z+cantilever_beam_side/2];
  
added_center_of_mass_wrt_foot{6} = ...
    [ -back_foot_sensor_offset_x + dataset06_cube_x_position_wrt_beam_begin + cube_side/2 ... 
      0.0 ...
      beam_upper_plane_wrt_foot_sensor_z + cantilever_beam_side + cube_com_wrt_base];

added_center_of_mass_wrt_foot{7} = [0.0 0.0 0.0];

added_center_of_mass_wrt_foot{8} = ...
   [ -back_foot_sensor_offset_x + cube_side/2 ... 
      0.0 ...
      beam_upper_plane_wrt_foot_sensor_z + cantilever_beam_side + cube_com_wrt_base];
    

for i = 1:n_datasets
    added_center_of_mass_wrt_leg{i} = added_center_of_mass_wrt_foot{i} - [ft_leg_position_wrt_ft_foot_x 0.0 ft_leg_position_wrt_ft_foot_z];
    added_first_moment_of_mass_wrt_leg{i} = added_mass(i)*added_center_of_mass_wrt_leg{i}; 
    added_first_moment_of_mass_wrt_foot{i} = added_mass(i)*added_center_of_mass_wrt_foot{i};
end

sampling_time = 0.01;

%% Loading data

%start building structure array
datasets = {};
n_datasets = length(datasets_names);
for i = 1:n_datasets
    datasets{i} = struct();
    datasets{i}.name = datasets_names{i};
    datasets{i}.drop_at_start = drop_at_start(i);
    datasets{i}.drop_at_end = drop_at_end(i);
end

for i = 1:n_datasets
    fprintf(['Loading and synchronizing dataset ' datasets{i}.name '\n']);
    [ft_foot, ft_leg, inertial, return_point] = ...
         import_synchronized_yarp_logs( sampling_time, datasets{i}.drop_at_start, datasets{i}.drop_at_end, ...
                               [datasets{i}.name '/ft_foot/data.log'], ...
                               [datasets{i}.name '/ft_leg/data.log'], ...
                               [datasets{i}.name '/inertial/data.log'], ...
                               [datasets{i}.name '/return_point/data.log']);
    
    assert(size(ft_foot,1) == size(ft_leg,1));
    assert(size(ft_foot,1) == size(return_point,1));
    datasets{i}.ft_foot_raw = ft_foot;
    datasets{i}.ft_leg_raw = ft_leg;
    datasets{i}.inertial_raw = inertial;
    datasets{i}.return_point = return_point;
end

sgolay_order = 3;
sgolay_window_len = 301;

for i = 1:n_datasets
    fprintf(['Smoothing and deriving dataset ' datasets{i}.name '\n']);
    datasets{i}.ft_foot_with_drift = sgolayvel(datasets{i}.ft_foot_raw(:,2:7),sgolay_order,sgolay_window_len);
    datasets{i}.ft_leg_with_drift =  sgolayvel(datasets{i}.ft_leg_raw(:,2:7),sgolay_order,sgolay_window_len);
    datasets{i}.acc = sgolayvel(datasets{i}.inertial_raw(:,5:7),sgolay_order,sgolay_window_len);
    %return point information is not filtered, but it should be cutted as
    %all the other signals
    window = sgolay_window_len;
    n_samples = size(datasets{i}.return_point,1);
    datasets{i}.return_point = datasets{i}.return_point((window+1)/2:(n_samples-(window/2)),2);
end

for i = 1:n_datasets
    datasets{i}.ft_foot = datasets{i}.ft_foot_with_drift;
    datasets{i}.ft_leg = datasets{i}.ft_leg_with_drift;
end

% % Remove offset drift in dataset, using return point position
% for i = 1:n_datasets
%     %save original version
%     %get the average of all return point
%     
%     % Compute return point samples
%     return_indices = find(datasets{i}.return_point == 1.0);
% 
%     % Find consecutive sequences of at least N return point samples
%     N = 10;
% 
%     % Find group of consecutive indices
%     return_indices_diff=diff(return_indices);
%     size(return_indices_diff)
% 
%     b = find([return_indices_diff' inf]>1);
%     sequence_lenghts = diff([0 b]);
% 
%     sequence_endpoints_indices = cumsum(sequence_lenghts);
% 
%     sequence_endpoints = return_indices(sequence_endpoints_indices);
% 
%     d = sequence_endpoints_indices+1;
% 
%     sequence_startpoints_indices = [1 d(1:(end-1))];
% 
%     sequence_startpoints = return_indices(sequence_startpoints_indices);
% 
%     % At this point we have the vector of sequence_startpoints +
%     % sequence_lengths 
%     n_return_points = length(sequence_startpoints);
%     
%     return_points_offsets_foot_raw = zeros(n_return_points,6);
%     return_points_offsets_leg_raw = zeros(n_return_points,6);
%     return_points_offsets_foot = zeros(n_return_points,6);
%     return_points_offsets_leg = zeros(n_return_points,6);
%     
%     
%     for j=1:n_return_points
%         return_points_offsets_foot_raw(j,:) = mean(datasets{i}.ft_foot_with_drift(sequence_startpoints(j):sequence_endpoints(j),:));
%         return_points_offsets_leg_raw(j,:) = mean(datasets{i}.ft_leg_with_drift(sequence_startpoints(j):sequence_endpoints(j),:));
%     end
%     
%     for j=1:n_return_points
%         return_points_offsets_foot(j,:) = return_points_offsets_foot_raw(j,:)-return_points_offsets_foot_raw(1,:);
%         return_points_offsets_leg(j,:) = return_points_offsets_leg_raw(j,:)-return_points_offsets_leg_raw(1,:);
%     end
%     
%     %removing drift
%     datasets{i}.ft_foot(1:sequence_startpoints(1),:) = datasets{i}.ft_foot_with_drift(1:sequence_startpoints(1),:);
%     datasets{i}.ft_leg(1:sequence_startpoints(1),:) = datasets{i}.ft_leg_with_drift(1:sequence_startpoints(1),:);
%     for j=1:n_return_points-1 
%         
%         datasets{i}.ft_foot(sequence_startpoints(j):sequence_endpoints(j),:) = ...
%             datasets{i}.ft_foot_with_drift(sequence_startpoints(j):sequence_endpoints(j),:) ...
%             -ones(sequence_endpoints(j)-sequence_startpoints(j)+1,1)*return_points_offsets_foot(j,:);
%         
%         datasets{i}.ft_foot(sequence_endpoints(j):sequence_startpoints(j+1),:) = ...
%             datasets{i}.ft_foot_with_drift(sequence_endpoints(j):sequence_startpoints(j+1),:) ...
%           -( ones(sequence_startpoints(j+1)-sequence_endpoints(j)+1,1)*return_points_offsets_foot(j,:) + ((return_points_offsets_foot(j+1,:)-return_points_offsets_foot(j,:))'*(0:1/(sequence_startpoints(j+1)-sequence_endpoints(j)):1))');
%         
%         datasets{i}.ft_leg(sequence_startpoints(j):sequence_endpoints(j),:) = ...
%             datasets{i}.ft_leg_with_drift(sequence_startpoints(j):sequence_endpoints(j),:) ...
%             -ones(sequence_endpoints(j)-sequence_startpoints(j)+1,1)*return_points_offsets_leg(j,:);
%         datasets{i}.ft_leg(sequence_endpoints(j):sequence_startpoints(j+1),:) = ...
%             datasets{i}.ft_leg_with_drift(sequence_endpoints(j):sequence_startpoints(j+1),:) - ...
%           -( ones(sequence_startpoints(j+1)-sequence_endpoints(j)+1,1)*return_points_offsets_leg(j,:) + ((return_points_offsets_leg(j+1,:)-return_points_offsets_leg(j,:))'*(0:1/(sequence_startpoints(j+1)-sequence_endpoints(j)):1))');
%     end
%     
%     last_interval_size = size(datasets{i}.ft_foot_with_drift(sequence_startpoints(n_return_points):end,:),1);
%     datasets{i}.ft_foot(sequence_startpoints(n_return_points):sequence_startpoints(n_return_points)+(last_interval_size-1),:) = ...
%         datasets{i}.ft_foot_with_drift(sequence_startpoints(n_return_points):end,:) ... 
%         -ones(last_interval_size,1)*return_points_offsets_foot(n_return_points,:);
%     
%     datasets{i}.ft_leg(sequence_startpoints(n_return_points):sequence_startpoints(n_return_points)+(last_interval_size-1),:) = ...
%         datasets{i}.ft_leg_with_drift(sequence_startpoints(n_return_points):end,:) ...
%         -ones(last_interval_size,1)*return_points_offsets_leg(n_return_points,:);
%     
% 
% end


%% Subspace estimation

square_side = ceil(sqrt(n_datasets));
figure
for i = 1:n_datasets
    datasets{i}.ft_foot_no_mean = datasets{i}.ft_foot-ones(size(datasets{i}.ft_foot,1),1)*mean(datasets{i}.ft_foot);
    datasets{i}.ft_foot_mean = mean(datasets{i}.ft_foot);
    subplot(square_side,2*square_side,2*i-1);
    [U_foot,S_ft_foot,V_foot]  = svd(datasets{i}.ft_foot_no_mean,'econ');
    bar((S_ft_foot));
    datasets{i}.ft_foot_projector = V_foot(:,1:3)';
    datasets{i}.ft_foot_projected = (V_foot(:,1:3)'*datasets{i}.ft_foot_no_mean')';
    
    datasets{i}.ft_leg_no_mean = datasets{i}.ft_leg-ones(size(datasets{i}.ft_leg,1),1)*mean(datasets{i}.ft_leg);
    datasets{i}.ft_leg_mean = mean(datasets{i}.ft_leg);
    
    subplot(square_side,2*square_side,2*i);
    [U_leg,S_ft_leg,V_leg] = svd(datasets{i}.ft_leg_no_mean,'econ');
    bar((S_ft_leg));
    datasets{i}.ft_leg_projector = V_leg(:,1:3)';
    datasets{i}.ft_leg_projected = (V_leg(:,1:3)'*datasets{i}.ft_leg_no_mean')';

end



% % compute 
% figure
% for i = 1:n_datasets
%     subplot(square_side,square_side,i);
%     plot(datasets{i}.ft_foot)
% end
% 
% % compute 
% figure
% for i = 1:n_datasets
%     subplot(square_side,square_side,i);
%     plot(datasets{i}.ft_leg)
% end

subsampling = 50;
norm_factor = 1000;

% normalize = @(x) x./norm_factor;
% denormalize = @(x) norm_factor.*x;

normalize = @(x) (x-ones(size(x,1),1)*mean(x))./(ones(size(x,1),1)*std(x));
normalize_isotropically = @(x) (x-ones(size(x,1),1)*mean(x))/mean(std(x));
% denormalize = @(x,y,z) (x.*(ones(size(x,1),1)*z)+ones(size(x,1),1)*y);

for i=1:n_datasets
    datasets{i}.ft_foot_projected_norm = normalize(datasets{i}.ft_foot_projected);
    datasets{i}.ft_leg_projected_norm = normalize(datasets{i}.ft_leg_projected);
end

for i=1:n_datasets
    fprintf(['Fitting foot ellipsoid for dataset ' datasets{i}.name '\n'])
    [datasets{i}.p_foot_norm,datasets{i}.foot_proj_norm_refitted] = ellipsoidfit_smart(datasets{i}.ft_foot_projected_norm(1:subsampling:end,:),datasets{i}.acc(1:subsampling:end,:));
    fprintf(['Fitting leg ellipsoid for dataset ' datasets{i}.name '\n']);
    [datasets{i}.p_leg_norm,datasets{i}.leg_proj_norm_refitted]   = ellipsoidfit_smart(datasets{i}.ft_leg_projected_norm(1:subsampling:end,:),datasets{i}.acc(1:subsampling:end,:));
    fprintf(['Fitting acc ellipsoid for dataset ' datasets{i}.name '\n']);
    datasets{i}.p_acc  = ellipsoidfit(datasets{i}.acc(1:subsampling:end,1),datasets{i}.acc(1:subsampling:end,2),datasets{i}.acc(1:subsampling:end,3));
end


%Plotting ellipsoid fitted in raw space
figure
for i = 1:n_datasets
    subplot(square_side,square_side,i);
    plot_ellipsoid_im(datasets{i}.p_foot_norm);    
    plot3_matrix(datasets{i}.ft_foot_projected_norm(1:subsampling:end,:))
    axis equal
end

figure
for i = 1:n_datasets
    subplot(square_side,square_side,i);
    plot_ellipsoid_im(datasets{i}.p_leg_norm);
    plot3_matrix(datasets{i}.ft_leg_projected_norm(1:subsampling:end,:))
    axis equal
end

figure
for i = 1:n_datasets
    subplot(square_side,square_side,i);
    plot_ellipsoid_im(datasets{i}.p_acc);
    plot3_matrix(datasets{i}.acc(1:subsampling:end,:))
    axis equal
end

%% Offset estimation
for i = 1:n_datasets
    [centers,ax] = ellipsoid_im2ex(datasets{i}.p_foot_norm);
    centers
    datasets{i}.center_foot_proj = denormalize2(centers',mean(datasets{i}.ft_foot_projected),std(datasets{i}.ft_foot_projected));
%     datasets{i}.foot_proj_norm_refitted_no_offset = datasets{i}.ft_foot_projected_norm_refitted - ones(size(datasets{i}.ft_foot_projected_norm_refitted,1),1)*centers';
%     datasets{i}.foot_proj_refitted_no_offset2 = denormalize2(datasets{i}.ft_foot_projected_norm_refitted,mean(datasets{i}.ft_foot_projected),std(datasets{i}.ft_foot_projected));
    datasets{i}.offset_foot = ((datasets{i}.ft_foot_projector')*datasets{i}.center_foot_proj')'+datasets{i}.ft_foot_mean;
    datasets{i}.foot_no_offset = datasets{i}.ft_foot - ones(size(datasets{i}.ft_foot,1),1)*datasets{i}.offset_foot;
%     datasets{i}.foot_proj_refitted_no_offset = datasets{i}.foot_proj_refitted - ones(size(datasets{i}.foot_proj_refitted,1),1)*datasets{i}.center_foot_proj;
    
    [centers,ax] = ellipsoid_im2ex(datasets{i}.p_leg_norm);
    datasets{i}.center_leg_proj = denormalize2(centers',mean(datasets{i}.ft_leg_projected),std(datasets{i}.ft_leg_projected));
    datasets{i}.offset_leg = ((datasets{i}.ft_leg_projector')*datasets{i}.center_leg_proj')'+datasets{i}.ft_leg_mean;
    datasets{i}.leg_no_offset = datasets{i}.ft_leg - ones(size(datasets{i}.ft_leg,1),1)*datasets{i}.offset_leg;
end

figure
for i=1:n_datasets
    subplot(square_side,square_side,i);
    plot3_matrix(datasets{i}.foot_no_offset*datasets{i}.ft_foot_projector');
    hold on
%     scatter3_matrix(datasets{i}.foot_proj_refitted_no_offset);
%     p_refitted_norm = ellipsoidfit(datasets{i}.foot_proj_refitted_no_offset2(:,1),datasets{i}.foot_proj_refitted_no_offset2(:,2),datasets{i}.foot_proj_refitted_no_offset2(:,3));
%     p_refitted = ellipsoidfit(datasets{i}.foot_proj_refitted_no_offset2(:,1),datasets{i}.foot_proj_refitted_no_offset2(:,2),datasets{i}.foot_proj_refitted_no_offset2(:,3));
%     plot_ellipsoid_im(p_refitted);
%     [centers,ax] = ellipsoid_im2ex(p_refitted);
    centers
end

%% Calibrate matrix
calibration_datasets = 1:4;
validation_datasets = 5:8;

acc_foot = zeros(40,1);
cov_foot = zeros(40,40);
acc_leg = zeros(40,1);
cov_leg = zeros(40,40);

for cal_dat = calibration_datasets
    for smpl = 1:size(datasets{cal_dat}.foot_no_offset,1)
        r_foot = datasets{cal_dat}.foot_no_offset(smpl,:);
        pi_known_foot = [added_mass(cal_dat) added_first_moment_of_mass_wrt_foot{cal_dat}];
        g = datasets{cal_dat}.acc(smpl,:);
        regr_foot = [ kron(r_foot,eye(6,6)) -kron(g,eye(6,6))*static_wrench_regressor ];
        kt_foot = static_wrench_matrix(pi_known_foot)*g';
        acc_foot = acc_foot + regr_foot'*kt_foot;
        cov_foot = cov_foot + regr_foot'*regr_foot;
        
        r_leg = datasets{cal_dat}.leg_no_offset(smpl,:);
        pi_known_leg = [added_mass(cal_dat) added_first_moment_of_mass_wrt_leg{cal_dat}];
        g = datasets{cal_dat}.acc(smpl,:);
        regr_leg = [ kron(r_leg,eye(6,6)) -kron(g,eye(6,6))*static_wrench_regressor ];
        kt_leg = static_wrench_matrix(pi_known_leg)*g';
        acc_leg = acc_leg + regr_leg'*kt_leg;
        cov_leg = cov_leg + regr_leg'*regr_leg;
        
        
        %raw_foot regr_foot = regr*known
        %raw_leg regr_leg = regr*known
    end
end

x_foot = pinv(cov_foot)*acc_foot;

x_leg = pinv(cov_leg)*acc_leg;
C_foot = reshape(x_foot(1:36),6,6);
C_leg = reshape(x_leg(1:36),6,6);
pi_unknown_foot = x_foot(37:40);
pi_unknown_leg = x_leg(37:40);

%% Validation plots

for val_dat = 1:n_datasets
%     for smpl = 1:size(datasets{cal_dat}.foot_no_offset,1)
%     end
    datasets{val_dat}.calibrated_foot_no_offset = datasets{val_dat}.foot_no_offset*C_foot';
    datasets{val_dat}.calibrated_leg_no_offset = datasets{val_dat}.leg_no_offset*C_leg';
    datasets{val_dat}.calibrated_leg_no_offset_original  = datasets{val_dat}.leg_no_offset*original_calibration_matrix_leg';
    datasets{val_dat}.calibrated_foot_no_offset_original = datasets{val_dat}.foot_no_offset*original_calibration_matrix_foot';
end

light_red   = [255.0 192.0 203.0]./255.0;
dark_red = [255.0 0 0]./255.0;
light_green = [144 238.0 144]./255.0;
dark_green  = [0 128.0 0]./255.0;

figure
for val_dat = 5:n_datasets
    fprintf(strcat('Plotting calibrated ellipsoids for foot force, dataset',datasets_names{val_dat}));

%   for smpl = 1:size(datasets{cal_dat}.foot_no_offset,1)
%   end
    hold on;
    subplot(2,2,val_dat-4);
    p_calib = ellipsoidfit_smart(datasets{val_dat}.calibrated_foot_no_offset(1:subsampling:end,1:3),datasets{val_dat}.acc(1:subsampling:end,:));
    ellipsoid_ecc(p_calib)
    hold on
    hold on
    hold on
    p_calib_original = ellipsoidfit_smart(datasets{val_dat}.calibrated_foot_no_offset_original(1:subsampling:end,1:3),datasets{val_dat}.acc(1:subsampling:end,:));
    ellipsoid_ecc(p_calib_original)
    plot_ellipsoid_im_color(p_calib_original,light_red);
    plot_ellipsoid_im_color(p_calib,light_green);
    plot3_matrix_color(datasets{val_dat}.calibrated_foot_no_offset_original(:,1:3),dark_red);
    plot3_matrix_color(datasets{val_dat}.calibrated_foot_no_offset(:,1:3),dark_green);

    view([45 10]);
    
    axis equal;
    axis([-11 11 -11 11 -11 11]);
    xlabel('f_x (N)')
    ylabel('f_y (N)')
    zlabel('f_z (N)')
end

figure
for val_dat = 5:n_datasets
    fprintf(strcat('Plotting calibrated ellipsoids for leg force',datasets_names{val_dat}));
%     for smpl = 1:size(datasets{cal_dat}.foot_no_offset,1)
%     end
    hold on;
    subplot(2,2,val_dat-4);
    p_calib_original = ellipsoidfit_smart(datasets{val_dat}.calibrated_leg_no_offset_original(1:subsampling:end,1:3),datasets{val_dat}.acc(1:subsampling:end,:));
      ellipsoid_ecc(p_calib_original)
    p_calib = ellipsoidfit_smart(datasets{val_dat}.calibrated_leg_no_offset(1:subsampling:end,1:3),datasets{val_dat}.acc(1:subsampling:end,:));

    ellipsoid_ecc(p_calib)
    plot_ellipsoid_im_color(p_calib_original,light_red);
    plot3_matrix_color(datasets{val_dat}.calibrated_leg_no_offset_original(:,1:3),dark_red);
    plot_ellipsoid_im_color(p_calib,light_green);
    plot3_matrix_color(datasets{val_dat}.calibrated_leg_no_offset(:,1:3),dark_green);
    view([45 10]);
    axis([-40 40 -60 60 -40 40]);
    axis equal
    xlabel('f_x (N)')
    ylabel('f_y (N)')
    zlabel('f_z (N)')
end

%% Validation inertial parameters estimation


for i=1:n_datasets
    fprintf(strcat('Estimating inertial parameters for dataset ',datasets_names{i}),'\n');
    % estimate inertial parameters for this dataset
    cov_par = zeros(4,4);
    kt_par_foot = zeros(4,1);
    kt_par_foot_old = zeros(4,1);
    kt_par_leg = zeros(4,1);
    kt_par_leg_old = zeros(4,1);
    for smpl = 1:size(datasets{i}.calibrated_leg_no_offset,1)
        inertial_param_regressor = inertial_parameter_static_regressor(datasets{i}.acc(smpl,:)');
        cov_par = cov_par + inertial_param_regressor'*inertial_param_regressor;
        kt_par_foot = kt_par_foot + inertial_param_regressor'*datasets{i}.calibrated_foot_no_offset(smpl,:)';
        kt_par_foot_old = kt_par_foot_old + inertial_param_regressor'*datasets{i}.calibrated_foot_no_offset_original(smpl,:)';
        kt_par_leg = kt_par_leg + inertial_param_regressor'*datasets{i}.calibrated_leg_no_offset(smpl,:)';
        kt_par_leg_old = kt_par_leg_old + inertial_param_regressor'*datasets{i}.calibrated_leg_no_offset_original(smpl,:)';
    end
    cov_var_inv = pinv(cov_par);
    datasets{i}.pi_foot = cov_var_inv*kt_par_foot;
    datasets{i}.pi_foot_old =  cov_var_inv*kt_par_foot_old;
    datasets{i}.pi_leg = cov_var_inv*kt_par_leg;
    datasets{i}.pi_leg_old = cov_var_inv*kt_par_leg_old;    
end

for i=1:n_datasets
    datasets{i}.pi_foot_added_known         = [added_mass(i) added_first_moment_of_mass_wrt_foot{i}];
    datasets{i}.pi_leg_added_known          = [added_mass(i) added_first_moment_of_mass_wrt_leg{i}];
    datasets{i}.pi_foot_added_estimated     = datasets{i}.pi_foot     - pi_unknown_foot;
    datasets{i}.pi_foot_added_estimated_old = datasets{i}.pi_foot_old - pi_unknown_foot;
    datasets{i}.pi_leg_added_estimated      = datasets{i}.pi_leg      - pi_unknown_leg;
    datasets{i}.pi_leg_added_estimated_old  = datasets{i}.pi_leg_old  - pi_unknown_leg;
end

figure
for i=1:n_datasets
    subplot(2,4,i);
    x = [1:3,6:8,11:13,16:18];
    y = [datasets{i}.pi_foot_added_known(1),datasets{i}.pi_foot_added_estimated(1),datasets{i}.pi_foot_added_estimated_old(1),...
         datasets{i}.pi_foot_added_known(2),datasets{i}.pi_foot_added_estimated(2),datasets{i}.pi_foot_added_estimated_old(2),...
         datasets{i}.pi_foot_added_known(3),datasets{i}.pi_foot_added_estimated(3),datasets{i}.pi_foot_added_estimated_old(3),...
         datasets{i}.pi_foot_added_known(4),datasets{i}.pi_foot_added_estimated(4),datasets{i}.pi_foot_added_estimated_old(4)];
     bar(x,y);
end

figure
for i=1:n_datasets
    subplot(2,4,i);
    x = [1:3,6:8,11:13,16:18];
    y = [datasets{i}.pi_leg_added_known(1),datasets{i}.pi_leg_added_estimated(1),datasets{i}.pi_leg_added_estimated_old(1),...
         datasets{i}.pi_leg_added_known(2),datasets{i}.pi_leg_added_estimated(2),datasets{i}.pi_leg_added_estimated_old(2),...
         datasets{i}.pi_leg_added_known(3),datasets{i}.pi_leg_added_estimated(3),datasets{i}.pi_leg_added_estimated_old(3),...
         datasets{i}.pi_leg_added_known(4),datasets{i}.pi_leg_added_estimated(4),datasets{i}.pi_leg_added_estimated_old(4)];
     bar(x,y);
end

% %% Diagnostics plots for debugging the local drift of the offset
% figure
% for i = 1:n_datasets
%     subplot(square_side,square_side,i);
%     plot3_matrix(datasets{i}.ft_leg_projected_norm(1:subsampling:end,:));
%     hold on
%     scatter3_matrix(datasets{i}.ft_leg_projected_norm(datasets{i}.return_point==1.0,:));
%     axis equal
% end
% 
% figure
% for i = 1:n_datasets
%     subplot(square_side,square_side,i);
%     plot3_matrix(datasets{i}.ft_foot_projected_norm(1:subsampling:end,:));
%     hold on
%     scatter3_matrix(datasets{i}.ft_foot_projected_norm(datasets{i}.return_point==1.0,:));
%     axis equal
% end
% 
% figure
% for i = 1:n_datasets
%     subplot(square_side,square_side,i);
%     plot3_matrix(datasets{i}.acc(1:subsampling:end,:));
%     hold on
%     scatter3_matrix(datasets{i}.acc(datasets{i}.return_point==1.0,:));
%     axis equal
% end
