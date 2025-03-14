#include <iostream>
#include <WinSock2.h>

#define PORT 8080

int main(){    
    char server_ip[100];
    std::cout << "Input Target Server IP: ";
    std::cin.getline(server_ip, sizeof(server_ip));

    // 初始化 WinSock
    WSADATA wsaData;
    if(WSAStartup(MAKEWORD(2,2), &wsaData)!=0){
        std::cerr << "WSAStartup failed: " << WSAGetLastError() << std::endl;
        return 1;
    }

    // 创建客户端Socket
    SOCKET clientSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(clientSock==INVALID_SOCKET){
        std::cerr << "socket() failed: " <<WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // 配置服务端地址
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;  
    serverAddr.sin_addr.s_addr = inet_addr(server_ip);  
    serverAddr.sin_port = htons(PORT);

    // 连接服务器
    if(connect(clientSock, (SOCKADDR*)&serverAddr, sizeof(serverAddr))==SOCKET_ERROR){
        std::cerr << "connect() failed: " << WSAGetLastError() << std::endl;
        closesocket(clientSock);
        WSACleanup();
        return 1;
    }

    std::cout << "Connected to server" << std::endl;

    // 通信循环
    char buffer[1024];
    while(true){
        std::cout << "Enter message (q to quit):";
        std::cin.getline(buffer, sizeof(buffer));
        if(strcmp(buffer, "q")==0) break;

        // 发送数据
        int sendSize = send(clientSock, buffer, strlen(buffer), 0);
        if(sendSize==SOCKET_ERROR){
            std::cerr << "send() failed: " << WSAGetLastError() << std::endl;
            break;
        }

        // 接收响应
        int recvSize = recv(clientSock, buffer, 1024, 0);
        if (recvSize>0){
            buffer[recvSize] = '\0';
            std::cout << "Server response: " << buffer << std::endl;
        }
    }

    // 清理资源
    closesocket(clientSock);
    WSACleanup();
    return 0;
}