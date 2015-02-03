#include <ros/ros.h>
#include "localize_drone.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "localizae_drone_node");
  ros::NodeHandle n;

  LocalizeDrone *node = new LocalizeDrone();
  node->init();

  image_transport::ImageTransport it(n);
  image_transport::CameraSubscriber sub_image = it.subscribeCamera("image_color", 1, &LocalizeDrone::imageCallback, node);
  image_transport::CameraSubscriber sub_depth = it.subscribeCamera("image_depth", 1, &LocalizeDrone::depthCallback, node);
  image_transport::Publisher pub_img = it.advertise("image_debug", 1);

  ros::Subscriber sub_cmd = n.subscribe("cmd", 1, &LocalizeDrone::cmdCallback, node);
  
  dynamic_reconfigure::Server<udrone::LocalizeDroneConfig> server;
  dynamic_reconfigure::Server<udrone::LocalizeDroneConfig>::CallbackType f;

  f = boost::bind(&LocalizeDrone::configCallback, node, _1, _2);
  server.setCallback(f);

  ros::Rate r(100);

  while (n.ok())
  {
    ros::spinOnce();
    r.sleep();
  }
  return 0;
}
