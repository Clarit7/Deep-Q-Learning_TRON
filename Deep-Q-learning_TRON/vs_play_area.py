from Net.ACNet import *
from tron.util import *
from ACKTR import Brain
import argparse

folderName = 'ex_saves2'

def main(args):
    iter=5000

    m = "1" if args.m is None else args.m
    if args.b == "2":
        b = 2
    elif args.b == "3":
        b = 3
    elif args.b == "4":
        b = 4
    elif args.b == "5":
        b = 5
    elif args.b == "6":
        b = 6
    elif args.b == "7":
        b = 7
    else:
        b = 1

    if args.c == "2":
        c = 2
    elif args.c == "3":
        c = 3
    elif args.c == "4":
        c = 4
    elif args.c == "5":
        c = 5
    elif args.c == "6":
        c = 6
    elif args.c == "7":
        c = 7
    else:
        c = 1

    if m == "2":
        actor_critic = Net8()  # 신경망 객체 생성
        global_brain = Brain(actor_critic, args, acktr=True)
        static_brain = None

        if b == 2 or b == 3:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-13_11_42-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_True-sep_True-8_40k_area_model.bak'))

            ac_static = NetStatic8()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-21_40_47-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-8_40k_pretrain.bak'))
            static_brain.actor_critic.eval()
        elif b == 4:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-19_25_42-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_False-sep_True-8_40k_greedy_model.bak'))
        elif b == 5:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.17-12_24_50-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_True-sep_False-8_40k_oneshot_model.bak'))
        elif b == 6:
            pass
        else:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.15-16_06_55-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_False-sep_True-8_40k_model.bak'))

            ac_static = NetStatic8()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-21_40_47-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-8_40k_pretrain.bak'))
            static_brain.actor_critic.eval()

        global_brain.actor_critic.eval()

        actor_critic2 = Net8()  # 신경망 객체 생성
        global_brain2 = Brain(actor_critic2, args, acktr=True)
        static_brain2 = None

        if c == 2 or c == 3:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-13_11_42-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_True-sep_True-8_40k_area_model.bak'))

            ac_static2 = NetStatic8()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-21_40_47-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-8_40k_pretrain.bak'))
            static_brain2.actor_critic.eval()
        elif c == 4:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-19_25_42-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_False-sep_True-8_40k_greedy_model.bak'))
        elif c == 5:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.17-12_24_50-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_True-sep_False-8_40k_oneshot_model.bak'))
        elif c == 6:
            pass
        else:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.15-16_06_55-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-area_False-sep_True-8_40k_model.bak'))

            ac_static2 = NetStatic8()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-21_40_47-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_8-8_40k_pretrain.bak'))
            static_brain2.actor_critic.eval()

        global_brain2.actor_critic.eval()
    elif m == "3":
        actor_critic = Net10()  # 신경망 객체 생성
        global_brain = Brain(actor_critic, args, acktr=True)
        static_brain = None

        if b == 2 or b == 3:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-13_12_03-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_True-10_40k_area_model.bak'))

            ac_static = NetStatic10()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.09-14_42_32-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-10_40k_pretrain.bak'))
            static_brain.actor_critic.eval()
        elif b == 4:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-19_26_09-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_False-sep_True-10_40k_greedy_model.bak'))
        elif b == 5:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.17-12_25_08-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_False-10_40k_oneshot_model.bak'))
        elif b == 6:
            pass
        else:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.15-18_25_39-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_False-sep_True-10_40k_model.bak'))

            ac_static = NetStatic10()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.09-14_42_32-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-10_40k_pretrain.bak'))
            static_brain.actor_critic.eval()

        global_brain.actor_critic.eval()

        actor_critic2 = Net10()  # 신경망 객체 생성
        global_brain2 = Brain(actor_critic2, args, acktr=True)
        static_brain2 = None

        if c == 2 or c == 3:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-13_12_03-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_True-10_40k_area_model.bak'))

            ac_static2 = NetStatic10()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.09-14_42_32-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-10_40k_pretrain.bak'))
            static_brain2.actor_critic.eval()

        elif c == 4:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.16-19_26_09-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_False-sep_True-10_40k_greedy_model.bak'))
        elif c == 5:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.17-12_25_08-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_True-sep_False-10_40k_oneshot_model.bak'))
        elif c == 6:
            pass
        else:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.15-18_25_39-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-area_False-sep_True-10_40k_model.bak'))

            ac_static2 = NetStatic10()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.09-14_42_32-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_10-10_40k_pretrain.bak'))
            static_brain2.actor_critic.eval()

        global_brain2.actor_critic.eval()
    else:
        actor_critic = Net6()  # 신경망 객체 생성
        global_brain = Brain(actor_critic, args, acktr=True)
        static_brain = None

        if b == 2 or b == 3:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.10-15_00_43-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_True-6_40k_area_model.bak'))

            ac_static = NetStatic6()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-15_49_13-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-6_40k_pretrain.bak'))
            static_brain.actor_critic.eval()
        elif b == 4:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.10-23_40_15-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_True-6_40k_greedy_model.bak'))
        elif b == 5:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.13-19_30_52-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_False-6_40k_oneshot_model.bak'))
        elif b == 6:
            pass
        elif b == 7:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.05-17_55_16-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_True-backtracking.bak'))
        else:
            global_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.09-18_04_55-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_False-sep_True-6_40k_model.bak'))

            ac_static = NetStatic6()
            static_brain = Brain(ac_static, args, acktr=True)
            static_brain.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-15_49_13-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-6_40k_pretrain.bak'))
            static_brain.actor_critic.eval()

        global_brain.actor_critic.eval()

        actor_critic2 = Net6()  # 신경망 객체 생성
        global_brain2 = Brain(actor_critic2, args, acktr=True)
        static_brain2 = None

        if c == 2 or c == 3:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.10-15_00_43-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_True-6_40k_area_model.bak'))

            ac_static2 = NetStatic6()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-15_49_13-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-6_40k_pretrain.bak'))
            static_brain2.actor_critic.eval()
        elif c == 4:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.10-23_40_15-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_True-6_40k_greedy_model.bak'))
        elif c == 5:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.13-19_30_52-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_False-6_40k_oneshot_model.bak'))
        elif c == 6:
            pass
        elif c == 7:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.05-17_55_16-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_True-sep_True-backtracking.bak'))
        else:
            global_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR-2021.03.09-18_04_55-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-area_False-sep_True-6_40k_model.bak'))

            ac_static2 = NetStatic6()
            static_brain2 = Brain(ac_static2, args, acktr=True)
            static_brain2.actor_critic.load_state_dict(torch.load(
                folderName + '/ACKTR_pretrain-2021.03.08-15_49_13-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-6_40k_pretrain.bak'))
            static_brain2.actor_critic.eval()

        global_brain2.actor_critic.eval()

    nullgame = 0
    p1_win = 0
    p2_win = 0

    len_sum = 0
    area_sum = 0
    area_count = 0

    """
    for _ in range(iter):
        game = make_static_game(True if b != 6 else False, True if c != 6 else False, "fair")
        p1_len, p1_area, p2_len, p2_area = game.main_loop(global_brain2.actor_critic, pop_up, None, global_brain.actor_critic,
                       static_brain=static_brain2.actor_critic if c == 1 or c == 2 else None,
                       static_brain2=static_brain.actor_critic if b == 1 or b == 2 else None,
                       oneshot_brain=global_brain2.actor_critic if c == 5 else None,
                       oneshot_brain2=global_brain.actor_critic if b == 5 else None,
                       end_separated=True, agent1=c, agent2=b)

        if game.winner is None:
            nullgame+=1
            print('draw')
        elif game.winner ==1:
            p2_win+=1
            print('p2_win')
        else:
            p1_win+=1
            print('p1_win')

        if p1_area != 0 and area_count < 10000:
            area_sum += p1_area
            len_sum += p1_len
            area_count += 1

        if p2_area != 0 and area_count < 10000:
            area_sum += p2_area
            len_sum += p2_len
            area_count += 1

        if area_count % 100 == 0:
            print("Aria count:{}".format(float(area_count)))
    """

    while area_count < 5000:
        game = make_game(True if b != 6 else False, True if c != 6 else False, "fair")
        p1_len, p1_area, p2_len, p2_area = game.main_loop(global_brain2.actor_critic, pop_up, None, global_brain.actor_critic,
                       static_brain=static_brain2.actor_critic if c == 1 or c == 2 else None,
                       static_brain2=static_brain.actor_critic if b == 1 or b == 2 else None,
                       oneshot_brain=global_brain2.actor_critic if c == 5 else None,
                       oneshot_brain2=global_brain.actor_critic if b == 5 else None,
                       end_separated=True, agent1=c, agent2=b)

        if game.winner is None:
            nullgame+=1
            print('draw')
        elif game.winner ==1:
            p2_win+=1
            print('p2_win')
        else:
            p1_win+=1
            print('p1_win')

        if p1_area != 0 and area_count < 10000:
            area_sum += p1_area
            len_sum += p1_len
            area_count += 1

        if p2_area != 0 and area_count < 10000:
            area_sum += p2_area
            len_sum += p2_len
            area_count += 1

        if area_count % 100 == 0:
            print("Aria count:{}".format(float(area_count)))

    print("Player 1:{} \n Player 2:{}\n ".format(p1_win,p2_win))
    print("Aria ratio:{}".format(float(len_sum / area_sum)))

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
