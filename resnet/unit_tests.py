import pickle as pkl
import os
import sys
import numpy as np
import torch
import resnet as resnet

'''
You can run this file to test your functions.
You should not need to modify this file, but you can if you want to get more specifics on errors.

To run this file, you should have a folder called seed=0_correct_function_output, this folder
should contain the correct outputs for each function. 

In theory you may pass a test with incorrect code although this is very unlikely.

When we grade, we will run these tests with a different seed, so you should not hard code any values.
'''

seed = 0
output_folder = './seed=0_correct_function_output/' # locally
os.makedirs(output_folder, exist_ok=True)
input_size = 100


def resnet_block_skip_connection():
    '''
    This test tests if the block handles the skip connection appropriately.
    '''
    print(f"{'-'*5} starting {sys._getframe().f_code.co_name} {'-'*5}")
    torch.manual_seed ( seed )

    F = torch.randn(128,16,32,32)
    X = torch.randn(128,16,32,32)

    block = resnet.Block(out_channels=16,stride=1)
    H_same_shape = block.skip_connection(F,X)

    F_diff = torch.randn(128,32,16,16)

    H_diff_shape = block.skip_connection(F_diff, X)
    
    outputs = dict(H_same_shape=H_same_shape,
                   H_diff_shape=H_diff_shape)

    with open(os.path.join(output_folder, 'resnet_block_skip_connection.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    for key in outputs.keys():
        assert torch.allclose(corr_outputs[key], outputs[key], atol=1e-05), f"{key} FAILED: Block skip connections does not match test"

    print("pass")


def resnet_block_forward_outshape():
    '''
    This test creates a Block instance with out_channels=16 and stride=1, 
    and feeds it a batch of 32 16x16 RGB images. It tests that the 
    output tensor has the expected shape.
    '''
    print(f"{'-'*5} starting {sys._getframe().f_code.co_name} {'-'*5}")
    torch.manual_seed ( seed )

    # Create a random input tensor with size (batch_size, channels, height, width)
    input_tensor = torch.randn(10, 16, 32, 32)  # For example, 10 images of size 32x32 with 16 channels

    # Create a ResNet block instance
    resnet_block = resnet.Block(out_channels=16, stride=1)

    # Set skip_connections to True or False based on your requirement
    skip_connections = True

    # Pass the input tensor through the ResNet block
    output_tensor = resnet_block(input_tensor, skip_connections=skip_connections)

    # Assert that the output tensor shape is as expected
    assert output_tensor.shape == (10, 16, 32, 32), "Output tensor shape does not match expected shape"
    print("pass")


def resnet_block_forward_test():
    print(f"{'-'*5} starting {sys._getframe().f_code.co_name} {'-'*5}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    block = resnet.Block(16, stride=1)
    X = torch.rand( [128, 16, 32, 32], dtype=torch.float32 )
    y_hat_stride1_no_skip = block.forward(X, skip_connections=False)
    y_hat_stride1_skip = block.forward(X, skip_connections=True)

    block = resnet.Block(16, stride=2)
    X = torch.rand( [128, 8, 16, 16], dtype=torch.float32 )
    y_hat_stride2_no_skip = block.forward(X, skip_connections=False)
    y_hat_stride2_skip = block.forward(X, skip_connections=True)

    outputs = dict(y_hat_stride1_no_skip=y_hat_stride1_no_skip,
                   y_hat_stride1_skip=y_hat_stride1_skip,
                   y_hat_stride2_no_skip=y_hat_stride2_no_skip,
                    y_hat_stride2_skip=y_hat_stride2_skip)
    
    with open(os.path.join(output_folder, 'resnet_block_forward.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)
    
    for key in outputs.keys():
        assert torch.allclose(corr_outputs[key], outputs[key], atol=1e-05), f"{key} FAILED: Block forward pass does not match test forward path"

    print("pass")



def resnet_forward_test():
    print(f"{'-'*5} starting {sys._getframe().f_code.co_name} {'-'*5}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    img_dim = 32
    n_channels = 3
    input_s = 128
    
    plainnet3 = resnet.Resnet(3, skip_connections=False) #20-layer cnn
    resnet3 = resnet.Resnet(3, skip_connections=True) #20-layer resnet
    inputs = torch.rand(input_s, n_channels, img_dim, img_dim, dtype=torch.float32) #random inputs
    yhat = resnet3.forward(inputs).detach().numpy() # feed it through the network to get predictions on one pass
    y_hat_plain = plainnet3.forward(inputs).detach().numpy()

    torch.no_grad()
    outputs = dict(yhat=yhat,
                   y_hat_plain=y_hat_plain
                    )
    
    with open(os.path.join(output_folder, 'resnet_forward.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)
    
    assert np.allclose(outputs['yhat'], corr_outputs['yhat'], atol=1e-06),  f'forward failed -- yhat outputs are not close enough'
    
    print("pass")


def resnet_tests():
    resnet_block_forward_outshape()
    resnet_block_skip_connection()
    resnet_block_forward_test()
    resnet_forward_test()
    # backward uses nn.Module's backward fn, so does not need testing

if __name__ == "__main__":
    resnet_tests()
    print('All tests passed!')
