import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import json

class SocketToROSPublisher(Node):
    def __init__(self):
        super().__init__('socket_to_ros_publisher')
        self.publisher_ = self.create_publisher(String, 'socket_data', 10)
        
    def start_server(self):
        host = '127.0.0.1'
        port = 65432
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            self.get_logger().info("Server started, waiting for connections...")
            while True:
                conn, addr = s.accept()
                self.get_logger().info(f"Connected by {addr}")
                self.handle_client(conn)

    def handle_client(self, conn):
        with conn:
            while True:
                data = conn.recv(2024)
                if not data:
                    break
                try:
                    received_objects = json.loads(data.decode('utf-8'))
                    self.get_logger().info(f"Received data: {received_objects}")
                    msg = String()
                    msg.data = json.dumps(received_objects)
                    self.publisher_.publish(msg)
                except:
                    self.get_logger().error("decode error")


def main(args=None):
    rclpy.init(args=args)
    socket_to_ros_publisher = SocketToROSPublisher()
    socket_to_ros_publisher.start_server()
    rclpy.spin(socket_to_ros_publisher)
    socket_to_ros_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
