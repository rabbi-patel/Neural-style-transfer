'''create main function
add arguments that are to be transformed form the directory
set your device'''


import argparse  #passes input arguments of file locations,model and things that are used repeatedly
import torch
from pathlib import Path
from utils.utils import *  
from torch.utils.data import DataLoader
from utils.models import*
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

def parse_arguments():
    parser=argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default='C:\\Users\\rabbi\\OneDrive\\Desktop\\Python_Files\\AI ML RESUME\\NST_CODE\\content_data',
                        help='location of content dataset')
    parser.add_argument('--style_dir',type=str,default='C:\\Users\\rabbi\\OneDrive\\Desktop\\Python_Files\\AI ML RESUME\\NST_CODE\\style_data',
                        help='location of style dataset')
    parser.add_argument('--vgg',type=str,default='C:\\Users\\rabbi\\OneDrive\\Desktop\\Python_Files\\AI ML RESUME\\NST_CODE\\vgg_normalised.pth',
                        help='location of pre-trained vgg')
    parser.add_argument('--experiment',type=str,default='experiment1',
                        help='name of experiment')
    
    #for transform we need argumens of image sizing and cropping
    parser.add_argument('--final_size',type=int,default=512,
                        help="Size of final image")
    parser.add_argument('--content_size',type=int,default=256,
                        help="Size of content image")
    parser.add_argument('--style_size',type=int,default=256,
                        help="Size of style image")
    parser.add_argument('--crop',type=bool)

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Laerning Rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    

    parser.add_argument('--epochs', type=int, default=6,
                        help='Number of epochs')
    
    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    


    return parser.parse_args()

def main():  #starting point of the program
    
    #1.pass arguments
    #how is it different from user input()=== it Reads values provided when the program starts by add.argument syntax.
    #Without add_argument(), if you tried to pass --epochs, Python wouldn't know what it means.
    args=parse_arguments()

    #2.setting your device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #3. make folders
    save_dir=Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True,parents=True) 
    #parents=true,creates experiment dir if the directory doesn't exist,exist_ok=true=doesn't throw filr error in case alreday present.

    #4.save argument values in text file
    with open(save_dir/'args.txt','w') as args_file:
        for key,value in vars(args).items():  #vars(args)-change arguments to dictionary values.
            args_file.write(f'{key}:{value}\n')
    
    #5.create dataset of the input data and use it by defining custom dataset in utils.py 
        #it is in ImageFolderDataset


    #6.we need to transform the images accordingly by using transform funtion and convert them into tensors.
    content_transform=get_transform(args.content_size,args.crop,args.final_size)
    style_transform=get_transform(args.style_size,args.crop,args.final_size)

    
    content_dataset=ImageFolderDataset(args.content_dir,content_transform)
    style_dataset=ImageFolderDataset(args.style_dir,style_transform)

    #7.datasets need to be loaded to data loaders
    '''I think you're at the stage where understanding how DataLoader works internally
    will make everything click. It essentially uses __len__() to determine the dataset size and 
    repeatedly calls __getitem__() to fetch individual samples and group them into batches. Once you see that workflow, 
    custom datasets become much easier to understand.'''

    content_dataloader=DataLoader(content_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True, #from gpu to cpu
                                  drop_last=True)   #drop_last=whatever remaining images after batching
    

    style_dataloader=DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)   #drop_last=whatever remaining images after batching
    
    print('Number of batches in content dataset: ', len(content_dataloader))
    print('Number of batches in style dataset: ', len(style_dataloader))

    #8. define models encoder and decoder
    encoder=VGGEncoder(args.vgg).to(device)
    decoder=Decoder().to(device)

    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)
    
    lr_schedular=optim.lr_scheduler.LambdaLR(
                                optimizer,
                                lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
                            )

#     Learning Rate

# 0.0010 ●
#        |
# 0.0008 |\
#        | \
# 0.0006 |  \
#        |   \
# 0.0004 |     \
#        |       \
# 0.0002 |         \
#        |___________\_________

#         0 20 40 60 80 100
#              Epoch
#As training progresses, the learning rate gets smaller.
# LambdaLR is a learning rate scheduler.
# lambda epoch: ... defines how the learning rate should change after each epoch.
# The formula returns a multiplier (not the learning rate itself).
# PyTorch multiplies the optimizer's initial learning rate by this multiplier every time you call lr_scheduler.step().

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path,map_location=device))
        
        optimizer.load_state_dict(torch.load(args.optimizer_path,map_location=device))

    # For example, if the initial learning rate is 0.001 and the lambda returns 0.5, the new learning rate becomes 0.0005.
   
    print("training")
    mse_loss=torch.nn.MSELoss()   #we will use square fn to learn the losses

    encoder.eval()

    running_loss=None
    running_closs=None
    running_sloss=None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))
                        # Suppose
                        # Content batches = 100
                        # Style batches = 120
                        # zip() stops when the shorter iterator finishes.therefore stops at 2 because c=2 batches.
        running_loss = 0
        running_closs = 0
        running_sloss = 0

        for content_batch,style_batch in progress_bar:
            #pass it to device
            content_batch=content_batch.to(device)
            style_batch=style_batch.to(device)

            #extract features of c an s through encoder
            c_feats=encoder(content_batch)
            s_feats=encoder(style_batch)

            #extrcated features ,pass them through ADAIN Layer
            t=adaptive_instance_normalisation(c_feats[-1],s_feats[-1])
            

            #normalised features of adain are saved in t ,pass it through decoder
            g=decoder(t)

            #g gives us an image,to compare we need fature map,pass i thorugh encoder again
            g_feats=encoder(g)

            #verify the content features are intact by comparing gen image fea an adain faetures
            # you can't compare with original bacuase it does not have the style feature like the adain output t has.
            # aing weights of ocntent so that original content value does not go away.
            loss_c=mse_loss(g_feats[-1],t)*args.content_weight

            loss_s=0
            #now we check style image but we have to go through every layer as the styles are distributed in evry layer.

            for g_f ,s_f in zip(g_feats,s_feats):
                g_mean,g_std=cal_mean_std(g_f)
                s_mean,s_std=cal_mean_std(s_f)
                loss_s+=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std)

            loss_s=loss_s*args.style_weight

            loss=loss_c+loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f}, Content Loss: {loss_c.item():4f}, Style Loss: {loss_s.item():4f}')
            #not fill your terminal it will keep upating for that one epocg for all the images.
        
            running_loss+=loss.item()
            running_closs+=loss_c.item()
            running_sloss+=loss_s.item()
            
        lr_schedular.step()

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)

        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss:{running_loss:4f}, Content Loss: {running_closs:4f}, Style Loss: {running_sloss:4f}')

       
        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)    


























if __name__=='__main__':
    main()