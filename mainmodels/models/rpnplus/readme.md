## RPN Model

### Model Architecture
```bash
content_rpn/conv1_1/filter:0 [5, 5, 1, 64] 0.078693
content_rpn/conv1_1/biases:0 [64] 0.0
content_rpn/conv1_2/filter:0 [5, 5, 64, 64] 5.1143
content_rpn/conv1_2/biases:0 [64] 0.0
content_rpn/conv2_1/filter:0 [5, 5, 64, 128] 10.2417
content_rpn/conv2_1/biases:0 [128] 0.0
content_rpn/conv2_2/filter:0 [5, 5, 128, 128] 20.4513
content_rpn/conv2_2/biases:0 [128] 0.0
content_rpn/conv3_1/filter:0 [3, 3, 128, 64] 3.66532
content_rpn/conv3_1/biases:0 [64] 0.0
content_rpn/conv3_2/filter:0 [3, 3, 64, 64] 1.84733
content_rpn/conv3_2/biases:0 [64] 0.0
content_rpn/conv3_3/filter:0 [3, 3, 64, 64] 1.85867
content_rpn/conv3_3/biases:0 [64] 0.0
content_rpn/conv4_1/filter:0 [3, 3, 64, 64] 1.83939
content_rpn/conv4_1/biases:0 [64] 0.0
content_rpn/conv4_2/filter:0 [3, 3, 64, 64] 1.85312
content_rpn/conv4_2/biases:0 [64] 0.0
content_rpn/conv4_3/filter:0 [3, 3, 64, 64] 1.85742
content_rpn/conv4_3/biases:0 [64] 0.0
content_rpn/gamma3:0 [] 1.0
content_rpn/gamma4:0 [] 0.5
content_rpn/conv_proposal_2/filter:0 [5, 2, 128, 128] 8.17913
content_rpn/conv_proposal_2/biases:0 [128] 0.0
content_rpn/conv_proposal_3/filter:0 [5, 2, 64, 256] 8.19745
content_rpn/conv_proposal_3/biases:0 [256] 0.0
content_rpn/conv_proposal_4/filter:0 [5, 2, 64, 256] 8.20255
content_rpn/conv_proposal_4/biases:0 [256] 0.0
content_rpn/conv_cls_score/filter:0 [1, 1, 640, 18] 0.576218
content_rpn/conv_cls_score/biases:0 [18] 0.0
content_rpn/conv_bbox_pred/filter:0 [1, 1, 640, 36] 1.16785
content_rpn/conv_bbox_pred/biases:0 [36] 0.0
```