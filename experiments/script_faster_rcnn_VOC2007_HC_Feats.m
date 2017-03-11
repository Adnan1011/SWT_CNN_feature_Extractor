function script_faster_rcnn_VOC2007_HC_Feats()
% script_faster_rcnn_VOC2007_HC_Feats()
% Faster rcnn training and testing using sophisticated trasnforms like
% Non-subsampled contourlet transform as feature extractors
% --------------------------------------------------------
% Built on top of:
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
% Author: Adnan Chaudhry
% Multimedia and Senors Lab (MSL)
% Georgia Tech
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.HC_Feats_for_Faster_RCNN_VOC2007;
% cache base
cache_base_proposal         = 'faster_rcnn_VOC2007_HC_Feats';
cache_base_fast_rcnn        = '';
% train/test data
dataset                     = [];
use_flipped                 = true;
dataset                     = Dataset.voc2007_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.voc2007_test(dataset, 'test', false);

%% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config('feat_stride', model.feat_stride);
conf_fast_rcnn              = fast_rcnn_config();
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model, true);
% generate anchors and pre-calculate output size of rpn network
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.rpn.cache_name, model.rpn.test_net_def_file);

%%  Phase 1 proposal
fprintf('\n***************\nPhase 1: Proposal \n***************\n');
% train
model.rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.rpn, opts.do_val, true);
% test
dataset.roidb_train  = cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.rpn, x, y, true), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test   = Faster_RCNN_Train.do_proposal_test(conf_proposal, model.rpn, dataset.imdb_test, dataset.roidb_test, true);

%%  Phase 2 Fast RCNN
fprintf('\n***************\nPhase 2: Fast RCNN\n***************\n');
% train
model.fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.fast_rcnn, opts.do_val, true);

%% Final test
fprintf('\n***************\nFinal test\n***************\n');
     
model.rpn.nms        = model.final_test.nms;
opts.final_mAP       = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.fast_rcnn, dataset.imdb_test, dataset.roidb_test, false, true);

% save final models, for outside tester
%Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file, true);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[3:5]);
end