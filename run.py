from control import joint_control,backpressure_control,only_perimeter_control

joint_controller=joint_control()
joint_controller.run()
joint_controller.RG()

only_perimeter_controller=only_perimeter_control(_with_logit_route_choice=True)
only_perimeter_controller.run()

only_perimeter_controller=only_perimeter_control()
only_perimeter_controller.run()

backpressure_controller=backpressure_control(_with_logit_route_choice=True)
backpressure_controller.run()

backpressure_controller=backpressure_control()
backpressure_controller.run()
