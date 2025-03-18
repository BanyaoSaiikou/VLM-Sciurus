#! /usr/bin/env python
# coding: utf-8
import os
import anthropic
import base64
import ast
import rospy
import time
from std_msgs.msg import String, Int32
#-------------------------------------------------------------------------------
image1_path = "/home/cwh/Desktop/expTF/0.jpg"
image2_path = "/home/cwh/Desktop/expTF/1.jpg"

image3_path = "/home/cwh/Desktop/expTF/2.jpg"
image4_path = "/home/cwh/Desktop/expTF/3.jpg"

#actionBEDONE = "open the bottom drawer"

#"Put <blue_cube> into bottom drawer."
#print("User Instructions:")

    







#-------------------------------------------------------------------------------


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="",
)




################################################################################
def generate_image_Descriptions(image1_path):
    # 读取图像并进行编码
    with open(image1_path, "rb") as image_file:
        image1_data = base64.b64encode(image_file.read()).decode("utf-8")
    

    # 创建消息
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="After the user uploads the image, generate a detailed description of the image. The description should include the main objects, scene, colors, spatial relationships, and any noteworthy details that can be observed. In addition, determine whether the drawer in the image is open or closed. Determine which drawer it is.Please describe the open and closed states of the bottom drawer.Please describe the open and closed states of the bottom space",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image1_data  # 使用实际的Base64数据
                        }
                    },


                ]
            }
        ]
    )
    print(message.content[0].text)
    return message.content[0].text  # 返回消息
################################################################################
######################################################    ######################################################
def generate_language_descriptioncheck(actionBEDONE, befor, after):
    # 读取图像并进行编码
    check_images_exist() 
    with open(befor , "rb") as image_file:
        #image1_data = base64.b64encode(image_file.read()).decode("utf-8")
        before_action_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    with open(after, "rb") as image_file:
        #image2_data = base64.b64encode(image_file.read()).decode("utf-8")
        after_action_image = base64.b64encode(image_file.read()).decode("utf-8")


    # 创建消息
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        temperature=0,
        system="You are an AI assistant with extensive visual knowledge and powerful reasoning abilities.\nYou will be provided with the following four types of information:\n1.<Instruction (Inst)>: The task that a human has requested the robot to perform.\n2.<Before-Action Image>: The image of what the robot is seeing before action.\n3.<Post-Action Image>: The image of what the robot is seeing after attempting the action, based on the given instruction.\n\n\nWhen the robot receives an <Instruction (Inst)>, it produces an action that changes the state of the objects in its field of vision. (<Image before action> changes to <Image after action>, <Image before action> does not exist again)\nHowever, due to various reasons, the robot may not always successfully execute the action.\n\nIn order to determine whether the action was successful, please compare <Instruction (Inst)>, <Before-Action Image>and <Post-Action Image> and judge whether the human action was successful.\n\nNote:\n    Provide clear and detailed descriptions.\n    Detailed observations and comparison images\n    If there are uncertainties or questions about the description, refrain from taking action and ask for clarification.\n    Please do not include content that does not exist in the image.\n    when Result is Success，Please also identify and point out if there is a significant difference in the range of the drawer opening.\n    Please do not infer or imagine anything that is not shown in the provided pictures.\n    Understand the situation of each image separately and request re-arrangement.\n\nOutput Structure:\nResult: [Success/Failure]\nReason: [Explanation of why the action was or was not successful]\nFinally, describe the location of each battery in the two pictures.\n\n",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please ignore the numbers in labeled.Please do not infer or imagine anything that is not shown in the provided pictures.\n\nYou will be provided with:\n\n\n\n1.<Instruction (Inst)>\n" + actionBEDONE + "\n\n2.<Before-Action Image>\n【"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": before_action_image  # 使用实际的Base64数据
                        }
                    },
                    {
                        "type": "text",
                        "text": "】\n3.<Post-Action Image>:\n【"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": after_action_image  # 使用实际的Base64数据
                        }
                    },
                    {
                        "type": "text",
                        "text": "】\n\n\n\n"
                    }
                ]
            }
        ]
    )
    return message.content[0].text  # 返回消息
    ######################################################     ######################################################
def generate_language_description(instructions, image1_path, image2_path, descriptions):
    # 读取图像并进行编码
    with open(image1_path, "rb") as image_file:
        image1_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    with open(image2_path, "rb") as image_file:
        image2_data = base64.b64encode(image_file.read()).decode("utf-8")
    



    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="Prompt: Language Description Generation for Robot Actions\n\n As an AI assistant with extensive visual knowledge and strong reasoning abilities, you will be provided with 1. Instructions (Inst), 2. the original image, and 3. the input image with boxes and labels. \n Your main task is to generate a language description of the input image based on the provided information. These descriptions are designed to help a robotic arm generate actions with one hand to perform the given instructions (Inst), while taking basic common sense into account.\n\nInstructions:\n\nHuman input consists of instructions (Inst) that prompt the robot to perform the corresponding actions to complete the given instructions. In addition, human input includes the original image and the input image with boxes and labels.\n\nYour role is to generate a language description of the input image, which is solely used to guide the robot in performing the provided instructions, including the required object state information, while considering basic common sense. (For example, when opening a door, both the door handle and the door need to be considered and described in the \"Input Image Description.\")\n\nThe description should reflect the current state of the objects in the input image, not necessarily the state after the implementation of the instruction (Inst). The state list provided below is used to express the spatial relationship between objects in the environment.\n\nThe robot actions must be limited to those specified in the \"Robot Action List\" provided below, considering basic common sense, such as the robot having only one hand and not being able to perform certain actions when it is already holding an object.\n\nIf there is uncertainty or doubt about the description, please do not take action and ask for clarification.\n\n\nSTATE LIST:\n\n    (<object>)is top_something(<something>): Object is located on top of <something> with direct contact\n    (<object>)is under_something(<something>): Object is located underneath <something> with direct contact\n    (<object>)is up_something(<something>): Object is located above <something> without direct contact\n    finish():Complete human instructions\n    (<object>)is down_something(<something>): Object is located below <something> with direct contact\n    (<object>)is left_to_something(<something>): Object is located left to the side of <something>\n    (<object>)is right_to_something(<something>): Object is located right to the side of <something>\n    (<object>)is up_left_to_something(<something>): Object is located in the upper left corner of the side <something>\n    (<object>)is up_right_to_something(<something>): Object is located in the upper right corner of the side <something>\n    (<object>)is down_left_to_something(<something>):Object is located in the lower left corner of the side of <something>\n    (<object>)is down_right_to_something(<something>): Object is located in the lower right corner of the side of <something>\n    (<object>)is inside_something(<something>): Object is located inside <something>\n    (<object>)is behind_something(<something>): Object is located behind <something>\n    (<object>)is in_front_of_something(<something>): Object is located in front of <something>\n    (<object>)is attached_to_something(<something>): Object is attached to <something>. For example, the door handle is attached to the door\n    (<object>)is closed()：object is closed.You cannot operate anything inside the object.\n    (<object>)is open()：object is opened.You can operate anything inside the object.\n\n    \n\nROBOT ACTION LIST:\n\n    move_hand(<object>): Move the robot hand from one position to another with/without grasping an object.\n    grasp_object(): Grasp an object. If an object is grasped, no other object can undergo any action from the \"ROBOT ACTION LIST\" until release_object() is executed.\n    release_object(): Release an object in the robot hand.\n    detach_from_plane(): Move the grabbed object from a state in which it is constrained by a plane to a state in which it is not constrained by any plane. For example, detach_from_plane() is used when a robot hand picks up an object on a table.\n    attach_to_plane(<object>): The opposite operation of detach_from_plane().\n    open_by_rotate(): Open something by rotating an object that is rotationally constrained by its environment along its rotation. For example, when opening a refrigerator, the refrigerator handle makes this motion. Also, when opening the lid of a plastic bottle, the lid makes this motion.\n    adjust_by_rotate(): Rotate an object that is rotationally constrained by its environment along its rotation. For example, when adjusting the temperature of a refrigerator, the temperature knob makes this motion.\n    close_by_rotate(): The opposite operation of open_by_rotate().\n    open_by_slide(): Moves an object that is translationally constrained in two axes from its environment along one unbounded axis. For example, when opening a sliding door or drawer, the handle makes this movement.\n    adjust_by_slide(): Slide an object that is translationally constrained in two axes from its environment along one unbounded axis. For example, when widening the gap between a sliding door and the wall, the handle makes this movement.\n    close_by_slide(): The opposite operation of open_by_slide().（The premise is that you get a grip.）\n    wipe_on_plane(): Move an object landing on a plane along two axes along that plane. For example, when wiping a window with a sponge, the sponge makes this motion.\n    robot_initial_pose(): The robot's posture is restored to a specified state. The current state of each object will not change.\n    put_to_temporary_location():Possible objects will be temporarily stored in temporary_location.\n\n    \"<object>\" in the ROBOT ACTION LIST can only use the information provided in the input image with boxes and labels.\n\nExample:\nDirective (Inst):\n\n\"Place the yellow, green, and red boxes on the table in the order of red, yellow, and green from top to bottom.\"\n\nInput Image Description:\n\n    (<yellow_box>).top_something(<table>)\n    (<green_box>).top_something(<table>)\n    (<green_box>).under_something(<blue_box>)\n    (<red_box>).top_something(<green_box>)\n\n\nSequence of actions required by ROBOT ACTION LIST:\n\n#Action.1: pick up the red_box\n------------------------------\n    move_hand(<red_box>)\n    grasp_object()\n    detach_from_plane()\n    robot_initial_pose() \n\n#Action.2: place the red_box on Temporary location\n------------------------------\n    put_to_temporary_location()\n    release_object() \n    robot_initial_pose() \n\n#Action.3: pick up the yellow_box\n------------------------------\n    move_hand(<yellow_box>)\n    grasp_object()\n    detach_from_plane()\n    robot_initial_pose() \n\n#Action.4: place the yellow_box on green_box\n------------------------------\n    move_hand(<green_box>)\n    attach_to_plane(<green_box>)\n    release_object()\n    robot_initial_pose() \n\n#Action.5: pick up the red_box\n------------------------------\n    move_hand(<red_box>)\n    grasp_object()\n    detach_from_plane()\n    robot_initial_pose() \n\n#Action.6: place the red_box on yellow_box\n------------------------------\n    move_hand(<yellow_box>)\n    attach_to_plane(<yellow_box>)\n    release_object() \n    robot_initial_pose()\nfinish()\n\nAdditional Requirements:\n\n    When describing the relationships or states of objects in the image, refer to the provided STATE LIST.\n    Consider basic common sense when generating sequences of robot actions.\n    If there are uncertainties or questions about the description, refrain from taking action and ask for clarification.\n    If there is a relationship between objects where one object is attached to another (e.g., <object_1> is attached_to_something(<object_2>)), both objects should be described in the output.\n    Considering that the robot only has one gripper.\n\n\nNote:\n    Please refer to the Inst. to complete the Inst. description, which describes the state of the object location and spatial relationships，open/close status.（Refer to the information in the boxes.）\n    Provide clear and detailed descriptions to guide robot actions effectively.\n    Consider the current state of objects in the input image, not just the desired state after implementing the directive.\n    In cases where multiple descriptions are necessary to fulfill the directive, include all relevant details to ensure accurate understanding and execution of the directive.\n    The noun in () cannot be in plural form.\n    when (<object>).is closed()，You cannot operate anything inside the object.\n    If the hand is currently grabbing an object, the \"grasp_object()\" operation cannot be performed before grasp_check(False) or put_to_temporary_location().\n    Possible objects be temporarily stored in temporary_location.\nRefer to the image descriptions to correctly output the open and closed status of the drawer.Note that 'drawer_top' refers to the top of the drawer and does not imply opening or closing it. Do not describe <drawer_top>.is_closed()，(<drawer_top>).is closed()，(<drawer_top>).is open() or <drawer_top>.is_open().",
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You will be provided with:\n1.instructions (Inst) :\n【\n   "+instructions+"\n】\n\n2.input original image :\n【"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image1_data
                    }
                },
                {
                    "type": "text",
                    "text": "】\n\n3.input image with box and label :\n【\n   "
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image2_data
                    }
                },
                {
                    "type": "text",
                    "text": "\n】\n\n4.detailed descriptions of the input image ：\n【\n"+descriptions+"\n】\n\n\n\n"
                    }
                ]
            }
        ]
    )
    return message.content[0].text
    
    
#print(message.content[0].text)
#print(type(message.content[0].text))


#----
def generate_action_sequence(image_description):
    message2 = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="For the input text, I would like you to identify the pattern from the examples I provide and output it in the desired format.Pay attention to the correspondence between "" \nPlease refer to the OUTPUT for the format. \nNo additional explanations are needed.\n\n\n\nINPUT：\n\nInput Image Description:\n\n(<blue_cube>).top_something(<cabinet>)\n(<cabinet>).top_something(<table>)\n(<bottom_drawer>).is down_something(<cabinet>)\n(<bottom_drawer>).is open()\n\nSequence of actions required by ROBOT ACTION LIST:\n\n#Action.1: pick up the blue_cube\n------------------------------\nmove_hand(<blue_cube>)\ngrasp_object()\ndetach_from_plane()\nrobot_initial_pose()\n\n#Action.2: place the blue_cube into the bottom_drawer \n------------------------------\nmove_hand(<bottom_drawer>)\nattach_to_plane(<bottom_drawer>)\nrelease_object()\nrobot_initial_pose()\n\n#Action.3: close the bottom_drawer\n------------------------------\nmove_hand(<bottom_drawer>)  \nclose_by_slide()\nrobot_initial_pose()\n\nOUTPUT：\n[\n[[0,\"pick up the blue_cube\"],[0,\"move_hand()\",\"blue_cube\"],[1,\"grasp_object()\",\"\"],[2,\"detach_from_plane()\",\"\"],[3,\"robot_initial_pose()\" ,\"\"]],\n\n[[1,\"place the blue_cube into the bottom_drawer\"],[0,\"move_hand()\",\"bottom_drawer\"],[1,\"attach_to_plane()\",\"bottom_drawer\"],[2,\"release_object()\",\"\"],[3,\"robot_initial_pose()\" ,\"\"]],\n\n[[2,\"place the blue_cube into the bottom_drawer\"],[0,\"move_hand()\",\"bottom_drawer\"],[1,\"close_by_slide()\",\"\"],[2,\"robot_initial_pose()\" ,\"\"]]\n]",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": image_description
                    }
                ]
            }
        ]
    )
    string_list = message2.content[0].text
    #print("!!!test",string_list,type(string_list))
    converted_list = ast.literal_eval(string_list)  # 将字符串转换为 Python 列表
    print("action：",converted_list[0][1])
    #print("!!!test2",converted_list,type(converted_list))
    return converted_list





def delete_files():
    file_paths = ["/home/cwh/Desktop/expTF/2.jpg", "/home/cwh/Desktop/expTF/3.jpg"]
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"{file_path} 已删除")
            else:
                print(f"{file_path} 不存在")
        except Exception as e:
            print(f"删除文件时出错: {e}")
def update_converted_list():
    # 在这里实现你的逻辑来更新 converted_list
    new_list = [
        # 新的动作序列
    ]
    return new_list
def check_images_exist():
    paths = [image1_path, image2_path, image3_path, image4_path]
    for path in paths:
        if not os.path.exists(path):
            print(f"警告: 图片不存在 {path}")
        else:
            print(f"图片存在: {path}")

# 发布机器人动作
def publish_robot_actions(converted_list):
    
    # 创建发布者
    pub = rospy.Publisher('robot_actions_topic', String, queue_size=10)
    check_pub = rospy.Publisher('user_check_topic', Int32, queue_size=10)  # 发布用户确认的数字
    rate = rospy.Rate(10)  # 10 Hz
    j = 0
    high_level_action = ""  # 初始化高层动作变量

    # 订阅 action2_topic 话题，更新 high_level_action
    def update_high_level_action(data):
        nonlocal high_level_action
        high_level_action = data.data
        print(f"Received high-level action: {high_level_action}")

    rospy.Subscriber('action3_topic3', String, update_high_level_action)  # 订阅 'action2_topic'

    # 新增的订阅者
    def callback(data):
        check = None
        nonlocal j
        j = 0
        k = 0
        if data.data == 1:  # 收到 1 后触发
            print("动作触发！")
            nonlocal converted_list
            rospy.sleep(5.0)  # 等待动作后图片保存
            if j == 0:
                rospy.sleep(3.0) 
                print("VLM2data:", high_level_action, image2_path, image4_path)  # 使用 high_level_action
                rospy.sleep(3.0) 
                #check_images_exist() 
                resultTF = generate_language_descriptioncheck(high_level_action, image2_path, image4_path)
            if j == 1:
                #rospy.sleep(3.0) 
                rospy.sleep(3.0) 
                print("VLM2data:", high_level_action, image2_path, image4_path)  # 使用 high_level_action
                resultTF = generate_language_descriptioncheck(high_level_action, image2_path, image4_path)
            
            print("VLM check action", resultTF)

            for i in range(50):
                if resultTF[i] == "S":
                    check = "T"
                    j = 1
                    print("check result:", check)
                    break
                elif resultTF[i] == "F":
                    check = "F"
                    j = 0
                    print("check result:", check)
                    break
            #rospy.sleep(3.0) 
            if check == "T-shan":  # 如果用户选择保持发布
                print("保持发布...")
                check_pub.publish(10)  # 发布数字 10
                rospy.sleep(0.5) ###删除前保留2秒
                delete_files()
            elif check == "F-shan":  # 如果用户选择不保持发布
                print("不保持发布，生成新的动作序列...")
                check_pub.publish(20)  # 发布数字 20 
                zzDescriptions=generate_image_Descriptions(image3_path)
                print("new state zzDescriptions:",zzDescriptions,image3_path)
                image_description = generate_language_description(user_instructions, image3_path, image4_path,zzDescriptions)
                print("Generated Image Description: \n", image_description)
                
                pubvlm1 = rospy.Publisher('prom_topic', String, queue_size=100)  # for UI
                rospy.sleep(0.1) 
                keyword = "Sequence of actions required by ROBOT ACTION LIST:"
                image_description_for_UI = image_description.split(keyword)[-1].strip()
                
                pubvlm1.publish(str(image_description_for_UI))

                converted_list = generate_action_sequence(image_description)
                
                print("Updated Action Sequence: \n", converted_list)
                rospy.sleep(0.5) ###删除前保留2秒
                delete_files()
            else:
                print("无效输入，请输入 'T' 或 'F'。")
                check = input("请输入 'T' 或 'F': ")
                
                #
                if check == "T":  # 如果用户选择保持发布
                    print("保持发布...")
                    check_pub.publish(10)  # 发布数字 10
                    rospy.sleep(0.5) ###删除前保留2秒
                    delete_files()
                elif check == "F":  # 如果用户选择不保持发布
                    print("不保持发布，生成新的动作序列...")
                    check_pub.publish(20)  # 发布数字 20 
                    zzDescriptions=generate_image_Descriptions(image3_path)
                    print("new state zzDescriptions:",zzDescriptions,image3_path)
                    image_description = generate_language_description(user_instructions, image3_path, image4_path,zzDescriptions)
                    print("Generated Image Description: \n", image_description)
                
                    pubvlm1 = rospy.Publisher('prom_topic', String, queue_size=100)  # for UI
                    rospy.sleep(0.1) 
                    keyword = "Sequence of actions required by ROBOT ACTION LIST:"
                    image_description_for_UI = image_description.split(keyword)[-1].strip()
                
                    pubvlm1.publish(str(image_description_for_UI))

                    converted_list = generate_action_sequence(image_description)
                
                    print("Updated Action Sequence: \n", converted_list)
                    rospy.sleep(0.5) ###删除前保留2秒
                    delete_files()
                
                

    rospy.Subscriber('start_action', Int32, callback)  # 订阅 'start_action' 话题

    while not rospy.is_shutdown():
        if pub.get_num_connections() > 0:  # 确保有订阅者
            for sequence in converted_list:
                pub.publish(str(converted_list))
                rate.sleep()


  
# 添加用于存储接收到的人类动作描述的全局变量
received_description = None
description_received_flag = False

# 添加处理接收到的人类动作描述的回调函数
def human_action_description_callback(msg):
    """处理接收到的人类动作描述并发送确认"""
    global received_description, description_received_flag
    
    # 获取描述
    received_description = msg.data
    description_received_flag = True
    
    # 打印接收到的描述
    rospy.loginfo("接收到人类动作描述: %s", received_description)
    
    # 发送确认消息
    confirmation = String()
    confirmation.data = "received"
    description_confirmation_pub.publish(confirmation)
    rospy.loginfo("已发送描述确认消息")
          
            

# 主函数
if __name__ == '__main__':
    try:
        # 获取用户指令
        rospy.init_node('robot_action_publisher', anonymous=True)
        pubzz = rospy.Publisher('instre_topic', String, queue_size=10)
        description_confirmation_pub = rospy.Publisher('/description_received', String, queue_size=10)
        rospy.Subscriber('/human_action_description', String, human_action_description_callback)
        Descriptions=generate_image_Descriptions(image1_path)
        
        
        # 如果需要从命令行输入指令（作为备选）
        use_received_description = True  # 设置为True使用接收到的描述，False使用命令行输入
        
        if use_received_description:
            # 等待接收人类动作描述
            rospy.loginfo("等待接收人类动作描述...")
            
            while not description_received_flag and not rospy.is_shutdown():
                rospy.sleep(0.1)
            
            # 使用接收到的描述作为指令
            user_instructions = received_description
            rospy.loginfo("使用接收到的描述作为指令: %s", user_instructions)
        else:
            # 使用命令行输入作为备选
            print("User Instructions:")
            user_instructions = input()
            print("User Instructions:", user_instructions)

        # 发布指令到UI
        pubzz.publish(str(user_instructions))
      


        # 调用函数生成图像描述
        image_description = generate_language_description(user_instructions, image1_path, image2_path, Descriptions)
        print("Generated Image Description: \n", image_description,type(image_description))


        pubvlm1 = rospy.Publisher('prom_topic', String, queue_size=100)#for UI
        rospy.sleep(0.1) 
        keyword = "Sequence of actions required by ROBOT ACTION LIST:"
        image_description_for_UI = image_description.split(keyword)[-1].strip()
        #print("------")
        #print("test1:",image_description_for_UI)
        #print("------")
        pubvlm1.publish(str(image_description_for_UI))



        # 根据图像描述生成动作序列
        action_sequence = generate_action_sequence(image_description)
        print("Generated Action Sequence: \n", action_sequence,len(action_sequence))
        #print("!!!",action_sequence,type(action_sequence))
        # 发布生成的机器人动作

        publish_robot_actions(action_sequence)

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"Error occurred: {e}")








