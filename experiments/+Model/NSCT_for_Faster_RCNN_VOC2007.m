function model = NSCT_for_Faster_RCNN_VOC2007(model)

% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;

%% Phase 1 RPN, Train RPN using NSCT features
model.rpn.solver_def_file                = fullfile(pwd, 'models', 'rpn_prototxts', 'NSCT', 'solver.prototxt');
model.rpn.test_net_def_file              = fullfile(pwd, 'models', 'rpn_prototxts', 'NSCT', 'test.prototxt');

% RPN test setting
model.rpn.nms.per_nms_topN              	= -1;
model.rpn.nms.nms_overlap_thres        	= 0.7;
model.rpn.nms.after_nms_topN           	= 2000;

%% Phase 2 Fast RCNN, train Fast RCNN layers using ROI's from RPN
model.fast_rcnn.solver_def_file          = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'NSCT', 'solver.prototxt');
model.fast_rcnn.test_net_def_file        = fullfile(pwd, 'models', 'fast_rcnn_prototxts', 'NSCT', 'test.prototxt');

%% final test
model.final_test.nms.per_nms_topN            	= 6000; % to speed up nms
model.final_test.nms.nms_overlap_thres       	= 0.7;
model.final_test.nms.after_nms_topN          	= 300;

end