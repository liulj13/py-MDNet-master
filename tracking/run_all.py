from run_tracker import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use the gpu 0
tb_100 = open('../../tracker_benchmark-master/data/tb_100.txt')  # open the file as a file
seq_list = tb_100.readlines()  # Read until EOF using readline() and return a list containing the lines thus read.
names = sorted([x.split('\t')[0].strip() for x in seq_list])  # x is each line, strip can remove the white space
# e.x. x is 'Basketball	IV, OCC, DEF, OPR, BC', after split and strip we get Basketball

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

for seq in names:
    args = argparse.ArgumentParser()  # create an ArgumentParser
    args.seq = seq
    args.display = False
    args.json = ''
    args.savefig = True

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    result_path = '../result/'+seq+'_pyMDNet.mat'
    # if (seq == 'David' or seq == 'Football1' or seq == 'Freeman3' or seq == 'Freeman4' or seq == 'Diving')==False:
    #     continue
    # if (seq != 'Tiger1'):
    #     continue
    # Run tracker
    if not os.path.exists(result_path):
        result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

        # Save result
        res = {}
        res['res'] = result_bb.round().tolist()
        res['type'] = 'rect'
        res['fps'] = fps
        # json.dump(res, open(result_path, 'w'), indent=2)  # indent display
        A = {'res': result_bb.round().tolist(), 'type': 'rect', 'len': len(img_list), 'fps': fps}  # structure
        results = np.zeros((1,), dtype=np.object)  # cell
        results[0] = A
        scipy.io.savemat(result_path, {'results': results})