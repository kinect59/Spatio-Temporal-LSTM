# Spatio-Temporal LSTM (ST-LSTM) with Trust Gates for 3D Human Action Recognition

```
@inproceedings{liu2016spatio,
  title={Spatio-Temporal LSTM with Trust Gates for 3D Human Action Recognition},  
  author={Liu, Jun and Shahroudy, Amir and Xu, Dong and Wang, Gang},  
  booktitle={ECCV},
  year={2016},
}
```

To run this code, make sure the following are installed:
* Torch7
* csvigo

To test the method on [NTU RGB+D dataset](https://github.com/shahroudy/NTURGB-D), 
please put the files "skl.csv", "descs.csv", and "training_testing_subjects.csv" in the folder "/data/". 

* "skl.csv" needs to include all the frames of all video samples. The row number is the frame number. The column number is 3x25x2, i.e., x1, y1, z1, x2, y2, z2...

* "descs.csv" is a description to index the frames strored in "skl.csv". The column number is the video sample number. The row index is respectively subject id, camera id, setup id, duplicate id, action id, starting frame id, and ending frame id. 

* "training_testing_subjects.csv" is the split of the train/test subjects for the [NTU RGB+D dataset](https://github.com/shahroudy/NTURGB-D).

### Related Publications:

J. Liu, A. Shahroudy, D. Xu, A. Kot, G. Wang, Skeleton-based action recognition using spatio-temporal LSTM network with trust gates, T-PAMI, 2017.

J. Liu, G. Wang, P. Hu, L. Duan, A. Kot, Global context-aware attention LSTM networks for 3D action recognition, CVPR, 2017.

The code is built on [char-rnn](https://github.com/karpathy/char-rnn).
Thanks to Andrej Karpathy‏.