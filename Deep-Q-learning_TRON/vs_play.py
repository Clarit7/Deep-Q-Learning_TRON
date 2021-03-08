from Net.ACNet import *
from tron.util import *
from ACKTR import Brain
import argparse

folderName = 'ex_saves'

def main(args):
    iter=1000

    m = "1" if args.m is None else args.m

    if args.b == "2":
        b = 2
    elif args.a == "3":
        b = 3
    elif args.a == "4":
        b = 4
    else:
        b = 1

    if args.c == "2":
        c = 2
    elif args.c == "3":
        c = 3
    elif args.c == "4":
        c = 4
    else:
        c = 1

    if m == "2":
        actor_critic = Net12()  # 신경망 객체 생성
        global_brain = Brain(actor_critic, args, acktr=True)
        static_brain = None

        if b == "2":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.02-14_55_52-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_True-sep_False-12_40k_area_model.bak'))
        elif b == "3":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.26-21_24_07-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_True-sep_True-12_40k_greedy_model.bak'))
        elif b == "4":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.27-21_03_08-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_True-sep_False-12_40k_oneshot_model.bak'))
        else:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.24-13_35_22-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_False-sep_True-12_40k_model.bak'))

            ac_static = NetStatic12()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.02.23-15_57_56-ent_0.15-pol_1.0-val_0.8-step_5-process_16-size_12-12_40k_model.bak'))
            static_brain.actor_critic.eval()

        global_brain.actor_critic.eval()

        actor_critic2 = Net12()  # 신경망 객체 생성
        global_brain2 = Brain(actor_critic2, args, acktr=True)
        static_brain2 = None

        if c == "2":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.02-14_55_52-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_True-sep_False-12_40k_area_model.bak'))
        elif c == "3":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.26-21_24_07-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_True-sep_True-12_40k_greedy_model.bak'))
        elif c == "4":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.27-21_03_08-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_True-sep_False-12_40k_oneshot_model.bak'))
        else:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.24-13_35_22-ent_0.1-pol_1.0-val_0.8-step_5-process_16-size_12-area_False-sep_True-12_40k_model.bak'))

            ac_static2 = NetStatic12()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.02.23-15_57_56-ent_0.15-pol_1.0-val_0.8-step_5-process_16-size_12-12_40k_model.bak'))
            static_brain2.actor_critic.eval()

        global_brain2.actor_critic.eval()
    elif m == "3":
        actor_critic = Net14()  # 신경망 객체 생성
        global_brain = Brain(actor_critic, args, acktr=True)
        static_brain = None

        if b == "2":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.02-14_56_20-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_True-sep_False-14_40k_area_model.bak'))
        elif b == "3":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.26-21_28_23-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_True-sep_True-14_40k_greedy_model.bak'))
        elif b == "4":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.27-21_03_35-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_True-sep_False-14_40k_oneshot_model.bak'))
        else:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.24-23_51_54-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_False-sep_True-14_40k_model.bak'))

            ac_static = NetStatic14()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.02.22-11_09_55-ent_0.1-pol_1.2-val_0.7-step_5-process_16-size_14-14_40k_model.bak'))
            static_brain.actor_critic.eval()

        global_brain.actor_critic.eval()

        actor_critic2 = Net14()  # 신경망 객체 생성
        global_brain2 = Brain(actor_critic2, args, acktr=True)
        static_brain2 = None

        if c == "2":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.02-14_56_20-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_True-sep_False-14_40k_area_model.bak'))
        elif c == "3":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.26-21_28_23-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_True-sep_True-14_40k_greedy_model.bak'))
        elif c == "4":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.27-21_03_35-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_True-sep_False-14_40k_oneshot_model.bak'))
        else:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.24-23_51_54-ent_0.15-pol_1.2-val_0.7-step_5-process_16-size_14-area_False-sep_True-14_40k_model.bak'))

            ac_static2 = NetStatic14()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.02.22-11_09_55-ent_0.1-pol_1.2-val_0.7-step_5-process_16-size_14-14_40k_model.bak'))
            static_brain2.actor_critic.eval()

        global_brain2.actor_critic.eval()
    else:
        actor_critic = Net10()  # 신경망 객체 생성
        global_brain = Brain(actor_critic, args, acktr=True)
        static_brain = None

        if b == "2":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.02-14_55_24-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_False-10_40k_area_model.bak'))
        elif b == "3":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.26-17_47_46-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_True-10_40k_greedy_model.bak'))
        elif b == "4":
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.27-21_01_51-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_False-10_40k_oneshot_model.bak'))
        else:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.23-18_57_50-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_False-sep_True-10_40k_model.bak'))

            ac_static = NetStatic10()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.02.22-05_09_11-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-10_40k_model.bak'))
            static_brain.actor_critic.eval()

        global_brain.actor_critic.eval()

        actor_critic2 = Net10()  # 신경망 객체 생성
        global_brain2 = Brain(actor_critic2, args, acktr=True)
        static_brain2 = None

        if c == "2":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.02-14_55_24-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_False-10_40k_area_model.bak'))
        elif c == "3":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.26-17_47_46-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_True-10_40k_greedy_model.bak'))
        elif c == "4":
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.27-21_01_51-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_False-10_40k_oneshot_model.bak'))
        else:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.02.23-18_57_50-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_False-sep_True-10_40k_model.bak'))

            ac_static2 = NetStatic10()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.02.22-05_09_11-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-10_40k_model.bak'))
            static_brain2.actor_critic.eval()

        global_brain2.actor_critic.eval()

    nullgame = 0
    p1_win = 0
    p2_win = 0

    for i in range(iter):
        game = make_game(True, True, "fair")
        game.main_loop(global_brain.actor_critic, pop_up, None, global_brain2.actor_critic,
                       static_brain=static_brain.actor_critic if b == 1 or b == 2 else None,
                       static_brain2=static_brain2.actor_critic if c == 1 or c == 2 else None,
                       oneshot_brain=global_brain.actor_critic if b == 4 else None,
                       oneshot_brain2=global_brain2.actor_critic if c == 4 else None,
                       end_separated=True, agent1=b, agent2=c)

        if game.winner is None:
            nullgame+=1
            print('draw')
        elif game.winner ==1:
            p1_win+=1
            print('p1_win')
        else:
            p2_win+=1
            print('p2_win')

    print("Player 1:{} \n Player 2:{}\n ".format(p1_win,p2_win))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', required=False, help='model structure number')
    parser.add_argument('-b', required=False, help='agent1 number')
    parser.add_argument('-c', required=False, help='agent2 number')
    parser.add_argument('-p', required=False, help='policy coefficient')
    parser.add_argument('-v', required=False, help='value coefficient')
    parser.add_argument('-a', required=False, help='get area instead of length')
    parser.add_argument('-e', required=False, help='True if end condition is separated')
    parser.add_argument('-u', required=False, help='unique string')

    args = parser.parse_args()

    main(args)
