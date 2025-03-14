#include <iostream>
#include <WinSock2.h>

#define PORT 8080
#define BUFFER_SIZE 1024

int main(){
    // 初始化 Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2,2), &wsaData)!=0){
        std::cerr << "WSAStartup Failed: " << WSAGetLastError() << std::endl;
        return 1;
    }

    // 创建监听socket
    SOCKET listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(listenSock==INVALID_SOCKET){
        std::cerr << "socket() failed: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return 1;
    }

    // 绑定地址
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddr.sin_port = htons(PORT);

    if(bind(listenSock, (SOCKADDR*)&serverAddr, sizeof(serverAddr))==SOCKET_ERROR){
        std::cerr << "bind() failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSock);
        WSACleanup();
        return 1;
    }

    // 开始监听
    if(listen(listenSock, SOMAXCONN) == SOCKET_ERROR){
        std::cerr << "listen() failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSock);
        WSACleanup();
        return 1;
    }

    std::cout << "Server listening on port: " << PORT << std::endl;

    // 接受客户端连接
    sockaddr_in clientAddr;
    int clientAddrLen = sizeof(clientAddr);
    SOCKET clientSock = accept(listenSock, (SOCKADDR*)&clientAddr, &clientAddrLen);
    if (clientSock == INVALID_SOCKET){
        std::cerr << "accept() failed: " << WSAGetLastError() << std::endl;
        closesocket(listenSock);
        WSACleanup();
        return 1;
    }

    std::cout << "Client Connected: " << inet_ntoa(clientAddr.sin_addr) << ":" << ntohs(clientAddr.sin_port) << std::endl;

    char buffer[BUFFER_SIZE];
    while(true){
        int recvSize = recv(clientSock, buffer, BUFFER_SIZE, 0);
        if (recvSize>0){
            buffer[recvSize] = '\0';
            std::cout << "Recived: " << buffer << std::endl;

            //回传数据
            send(clientSock, buffer, recvSize, 0);
        }
        else if(recvSize == 0){
            std::cout << "Connection closed" << std::endl;
            break;
        }
        else{
            std::cerr << "recv() failed: " << WSAGetLastError() << std::endl;
            break;
        }
    }

    // 清理资源
    closesocket(clientSock);
    closesocket(listenSock);
    WSACleanup(); 
    return 0;

}