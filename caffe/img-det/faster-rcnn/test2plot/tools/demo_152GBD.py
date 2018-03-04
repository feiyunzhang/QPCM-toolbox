#!/usr/bin/env python

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import random

CLASSES = ('__background__',
           'Person', 'Clothing', 'Man', 'Face', 'Tree', 'Plant', 'Woman', 'Vehicle', 'Building', 'Land_vehicle', 'Footwear', 'Girl', 'Animal', 'Car', 'Food', 'Wheel', 'Flower', 'Furniture', 'Window', 'House', 'Boy', 'Fashion_accessory', 'Table', 'Glasses', 'Suit', 'Auto_part', 'Bird', 'Sports_equipment', 'Dress', 'Dog', 'Carnivore', 'Human_body', 'Jeans', 'Musical_instrument', 'Drink', 'Boat', 'Hair', 'Tire', 'Head', 'Cat', 'Watercraft', 'Chair', 'Bike', 'Tower', 'Mammal', 'Skyscraper', 'Arm', 'Toy', 'Sculpture', 'Invertebrate', 'Microphone', 'Poster', 'Insect', 'Guitar', 'Nose', 'Hat', 'Tableware', 'Door', 'Bicycle_wheel', 'Sunglasses', 'Baked_goods', 'Eye', 'Dessert', 'Mouth', 'Aircraft', 'Airplane', 'Train', 'Jacket', 'Street_light', 'Hand', 'Snack', 'Helmet', 'Trousers', 'Bottle', 'Houseplant', 'Horse', 'Desk', 'Palm_tree', 'Vegetable', 'Fruit', 'Leg', 'Book', 'Fast_food', 'Beer', 'Flag', 'Drum', 'Bus', 'Truck', 'Ball', 'Tie', 'Flowerpot', 'Goggles', 'Motorcycle', 'Picture_frame', 'Shorts', 'Sports_uniform', 'Moths_and_butterflies', 'Shelf', 'Shirt', 'Fish', 'Rose', 'Licence_plate', 'Couch', 'Weapon', 'Laptop', 'Wine_glass', 'Van', 'Wine', 'Duck', 'Bicycle_helmet', 'Butterfly', 'Swimming_pool', 'Ear', 'Office', 'Camera', 'Stairs', 'Reptile', 'Football', 'Cake', 'Mobile_phone', 'Sun_hat', 'Coffee_cup', 'Christmas_tree', 'Computer_monitor', 'Helicopter', 'Bench', 'Castle', 'Coat', 'Porch', 'Swimwear', 'Cabinetry', 'Tent', 'Umbrella', 'Balloon', 'Billboard', 'Bookcase', 'Computer_keyboard', 'Doll', 'Dairy', 'Bed', 'Fedora', 'Seafood', 'Fountain', 'Traffic_sign', 'Hiking_equipment', 'Television', 'Salad', 'Bee', 'Coffee_table', 'Cattle', 'Marine_mammal', 'Goose', 'Curtain', 'Kitchen_&_dining', 'Home_appliance', 'Marine_invertebrates', 'Countertop', 'Office_supplies', 'Luggage_and_bags', 'Lighthouse', 'Cocktail', 'Maple', 'Saucer', 'Paddle', 'Bronze_sculpture', 'Beetle', 'Box', 'Necklace', 'Monkey', 'Whiteboard', 'Plumbing_fixture', 'Kitchen_appliance', 'Plate', 'Coffee', 'Deer', 'Surfboard', 'Turtle', 'Tool', 'Handbag', 'Football_helmet', 'Canoe', 'Cart', 'Scarf', 'Beard', 'Drawer', 'Cowboy_hat', 'Clock', 'Convenience_store', 'Sandwich', 'Traffic_light', 'Spider', 'Bread', 'Squirrel', 'Vase', 'Rifle', 'Cello', 'Pumpkin', 'Elephant', 'Lizard', 'Mushroom', 'Baseball_glove', 'Juice', 'Skirt', 'Skull', 'Lamp', 'Musical_keyboard', 'High_heels', 'Falcon', 'Ice_cream', 'Mug', 'Watch', 'Boot', 'Ski', 'Taxi', 'Sunflower', 'Pastry', 'Tap', 'Bowl', 'Glove', 'Parrot', 'Eagle', 'Tin_can', 'Platter', 'Sandal', 'Violin', 'Penguin', 'Sofa_bed', 'Frog', 'Chicken', 'Lifejacket', 'Sink', 'Strawberry', 'Bear', 'Muffin', 'Swan', 'Candle', 'Pillow', 'Owl', 'Kitchen_utensil', 'Dragonfly', 'Tortoise', 'Mirror', 'Lily', 'Pizza', 'Coin', 'Cosmetics', 'Piano', 'Tomato', 'Chest_of_drawers', 'Teddy_bear', 'Tank', 'Squash', 'Lion', 'Brassiere', 'Sheep', 'Spoon', 'Dinosaur', 'Tripod', 'Tablet_computer', 'Rabbit', 'Skateboard', 'Snake', 'Shellfish', 'Sparrow', 'Apple', 'Goat', 'French_fries', 'Lipstick', 'Studio_couch', 'Hamburger', 'Tea', 'Telephone', 'Baseball_bat', 'Bull', 'Headphones', 'Lavender', 'Parachute', 'Cookie', 'Tiger', 'Pen', 'Racket', 'Fork', 'Bust', 'Miniskirt', 'Sea_lion', 'Egg', 'Saxophone', 'Giraffe', 'Waste_container', 'Snowboard', 'Wheelchair', 'Medical_equipment', 'Antelope', 'Harbor_seal', 'Toilet', 'Shrimp', 'Orange', 'Cupboard', 'Wall_clock', 'Pig', 'Nightstand', 'Bathroom_accessory', 'Grape', 'Dolphin', 'Lantern', 'Trumpet', 'Tennis_racket', 'Crab', 'Sea_turtle', 'Cannon', 'Accordion', 'Door_handle', 'Lemon', 'Foot', 'Mouse', 'Wok', 'Volleyball', 'Pasta', 'Earrings', 'Banana', 'Ladder', 'Backpack', 'Crocodile', 'Roller_skates', 'Scoreboard', 'Jellyfish', 'Sock', 'Camel', 'Plastic_bag', 'Caterpillar', 'Sushi', 'Whale', 'Leopard', 'Barrel', 'Fireplace', 'Stool', 'Snail', 'Candy', 'Rocket', 'Cheese', 'Billiard_table', 'Mixing_bowl', 'Bowling_equipment', 'Knife', 'Loveseat', 'Hamster', 'Mouse', 'Shark', 'Teapot', 'Trombone', 'Panda', 'Zebra', 'Mechanical_fan', 'Carrot', 'Cheetah', 'Gondola', 'Bidet', 'Jaguar', 'Ladybug', 'Crown', 'Snowman', 'Bathtub', 'Table_tennis_racket', 'Sombrero', 'Brown_bear', 'Lobster', 'Refrigerator', 'Oyster', 'Handgun', 'Oven', 'Kite', 'Rhinoceros', 'Fox', 'Light_bulb', 'Polar_bear', 'Suitcase', 'Broccoli', 'Otter', 'Mule', 'Woodpecker', 'Starfish', 'Kettle', 'Jet_ski', 'Window_blind', 'Raven', 'Grapefruit', 'Chopsticks', 'Tart', 'Watermelon', 'Cucumber', 'Infant_bed', 'Missile', 'Gas_stove', 'Bathroom_cabinet', 'Beehive', 'Alpaca', 'Doughnut', 'Hippopotamus', 'Ipod', 'Kangaroo', 'Ant', 'Bell_pepper', 'Goldfish', 'Ceiling_fan', 'Shotgun', 'Barge', 'Potato', 'Jug', 'Microwave_oven', 'Bat', 'Ostrich', 'Turkey', 'Sword', 'Tennis_ball', 'Pineapple', 'Closet', 'Stop_sign', 'Taco', 'Pancake', 'Hot_dog', 'Organ', 'Rays_and_skates', 'Washing_machine', 'Waffle', 'Snowplow', 'Koala', 'Honeycomb', 'Sewing_machine', 'Horn', 'Frying_pan', 'Seat_belt', 'Zucchini', 'Golf_cart', 'Pitcher', 'Fire_hydrant', 'Ambulance', 'Golf_ball', 'Tiara', 'Raccoon', 'Belt', 'Corded_phone', 'Swim_cap', 'Red_panda', 'Asparagus', 'Scissors', 'Limousine', 'Filing_cabinet', 'Bagel', 'Wood-burning_stove', 'Segway', 'Ruler', 'Bow_and_arrow', 'Balance_beam', 'Kitchen_knife', 'Cake_stand', 'Banjo', 'Flute', 'Rugby_ball', 'Dagger', 'Dog_bed', 'Cabbage', 'Picnic_basket', 'Peach', 'Submarine_sandwich', 'Pear', 'Lynx', 'Pomegranate', 'Shower', 'Blue_jay', 'Printer', 'Hedgehog', 'Coffeemaker', 'Worm', 'Drinking_straw', 'Remote_control', 'Radish', 'Canary', 'Seahorse', 'Wardrobe', 'Toilet_paper', 'Centipede', 'Croissant', 'Snowmobile', 'Burrito', 'Porcupine', 'Cutting_board', 'Dice', 'Harpsichord', 'Perfume', 'Drill', 'Calculator', 'Willow', 'Pretzel', 'Guacamole', 'Popcorn', 'Harp', 'Towel', 'Mixer', 'Digital_clock', 'Alarm_clock', 'Artichoke', 'Milk', 'Common_fig', 'Power_plugs_and', 'Paper_towel', 'Blender', 'Scorpion', 'Stretcher', 'Mango', 'Magpie', 'Isopod', 'Personal_care', 'Unicycle', 'Punching_bag', 'Envelope', 'Scale', 'Wine_rack', 'Submarine', 'Cream', 'Chainsaw', 'Cantaloupe', 'Serving_tray', 'Food_processor', 'Dumbbell', 'Jacuzzi', 'Slow_cooker', 'Syringe', 'Dishwasher', 'Tree_house', 'Briefcase', 'Stationary_bicycle', 'Oboe', 'Treadmill', 'Binoculars', 'Bench', 'Cricket_ball', 'Salt_and_pepper', 'Squid', 'Light_switch', 'Toothbrush', 'Spice_rack', 'Stethoscope', 'Winter_melon', 'Ladle', 'Flashlight')


def demo(net, data_dir, imgfile, out_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(data_dir, imgfile)
    im = cv2.imread(im_file)

    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    scores = np.squeeze(scores)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.12
    NMS_THRESH = 0.3
    color_white = (0, 0, 0)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=4)
            cv2.putText(im, '%s %.3f' % (cls, score), (bbox[0], bbox[1] + 15),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  

    args = parse_args()
    cfg.GPU_ID = args.gpu_id
    prototxt = '../demo/model/deploy.prototxt'
    caffemodel = '../demo/model/test.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)


    data_dir = '../demo/images/'
    out_dir = '../demo/output/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    imgpath = os.listdir(data_dir)
    for imgfile in imgpath:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(imgfile)
        im = demo(net, data_dir, imgfile, out_dir)
        cv2.imwrite('../demo/output/' + imgfile, im)
