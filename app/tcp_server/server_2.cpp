#include <iostream>
#include <thread>
#include <vector>
#include <WinSock2.h>

#define PORT 8080
#define BUFFER_SIZE 1024

void client_handler(SOCKET client_socket){
    
    char buffer[BUFFER_SIZE];
    while (true){
        int recv_size = recv(client_socket, buffer, BUFFER_SIZE, 0);
        if(recv_size>0){
            buffer[recv_size] = '\0';
            std::cout << "Recived: " << buffer << std::endl;
            send(client_socket, buffer, recv_size, 0);
        }else if(recv_size==0){
            std::cout << "Client Disconnected" << std::endl;  
            break;                  
        }else{
            std::cerr << "Recv Error: " <<WSAGetLastError() << std::endl;
            break;
        }
    }

    closesocket(client_socket);
}

int main(){
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);

    // 创建 监听socket
    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    // htons  host-to-network for type 'short'  将端口号转为大端序
    // htonl  host-to-network for type 'long'   将IP地址转为大端序
    sockaddr_in server_addr{AF_INET, htons(PORT), htonl(INADDR_ANY)};  

    bind(listen_sock, (SOCKADDR*)&server_addr, sizeof(server_addr));
    listen(listen_sock, SOMAXCONN);
    std::cout << "Server listening on port: " << PORT << std::endl;

    std::vector<std::thread> threads;
    while(true){
        sockaddr_in clientAddr;
        int clientAddrLen = sizeof(clientAddr);
        // SOCKET client_socket = accept(listen_sock, nullptr, nullptr); 
        SOCKET client_socket = accept(listen_sock, (SOCKADDR*)&clientAddr, &clientAddrLen); 
        std::cout << "Client Connected: " << inet_ntoa(clientAddr.sin_addr) << ":" << ntohs(clientAddr.sin_port) << std::endl;

        if(client_socket==INVALID_SOCKET){
            std::cerr << "accept error: " << WSAGetLastError() << std::endl;
            continue;
        }

        threads.emplace_back(client_handler, client_socket);
        threads.back().detach();  // 设置线程分离  独立运行 即使主线程结束 分离的子线程仍会继续执行

        std::cout << "New Client connected. Active threads: " << threads.size() << std::endl;
    }

    closesocket(listen_sock);
    WSACleanup();
    return 0;
}