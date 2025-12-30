#include <iostream>
#include <chrono>
#include <omp.h>

// Односвязный список 
struct Node {
    int val;
    Node* next;
    Node(int v) : val(v), next(nullptr) {}
};

struct SinglyList {
    Node* head = nullptr;

    void push_front(int v) { // добавить в начало
        Node* n = new Node(v);
        n->next = head;
        head = n;
    }

    bool find(int v) const { // поиск
        Node* cur = head;
        while (cur) {
            if (cur->val == v) return true;
            cur = cur->next;
        }
        return false;
    }

    bool remove(int v) { // удалить первое совпадение
        Node* cur = head;
        Node* prev = nullptr;

        while (cur) {
            if (cur->val == v) {
                if (prev) prev->next = cur->next;
                else head = cur->next;
                delete cur;
                return true;
            }
            prev = cur;
            cur = cur->next;
        }
        return false;
    }

    void clear() { // очистка
        Node* cur = head;
        while (cur) {
            Node* nxt = cur->next;
            delete cur;
            cur = nxt;
        }
        head = nullptr;
    }

    ~SinglyList() { clear(); }
};

// Стек (на списке) 
struct Stack {
    Node* top = nullptr;

    void push(int v) { // добавить
        Node* n = new Node(v);
        n->next = top;
        top = n;
    }

    bool isEmpty() const { // пустой?
        return top == nullptr;
    }

    bool pop(int& out) { // снять
        if (isEmpty()) return false;
        Node* t = top;
        out = t->val;
        top = t->next;
        delete t;
        return true;
    }

    void clear() {
        while (top) {
            Node* nxt = top->next;
            delete top;
            top = nxt;
        }
    }

    ~Stack() { clear(); }
};

// Очередь 
struct Queue {
    Node* front = nullptr;
    Node* back = nullptr;

    bool isEmpty() const { // пустая?
        return front == nullptr;
    }

    void push_back(int v) { // добавить в конец
        Node* n = new Node(v);
        if (!back) {
            front = back = n;
        } else {
            back->next = n;
            back = n;
        }
    }

    bool pop_front(int& out) { // удалить из начала
        if (isEmpty()) return false;
        Node* f = front;
        out = f->val;
        front = f->next;
        if (!front) back = nullptr;
        delete f;
        return true;
    }

    void clear() {
        Node* cur = front;
        while (cur) {
            Node* nxt = cur->next;
            delete cur;
            cur = nxt;
        }
        front = back = nullptr;
    }

    ~Queue() { clear(); }
};

// Тест производительности 
long long add_list_seq(SinglyList& L, int n) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        L.push_front(i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

long long add_list_omp(SinglyList& L, int n) {
    auto t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
#pragma omp critical
        {
            L.push_front(i);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

long long add_queue_seq(Queue& Q, int n) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        Q.push_back(i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

long long add_queue_omp(Queue& Q, int n) {
    auto t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
#pragma omp critical
        {
            Q.push_back(i);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

int main() {
    std::cout << "Part 2: Data structures + OpenMP\n\n";

    SinglyList L;
    L.push_front(10);
    L.push_front(20);
    std::cout << "List find 10: " << (L.find(10) ? "YES" : "NO") << "\n";
    std::cout << "List remove 10: " << (L.remove(10) ? "OK" : "NO") << "\n";
    std::cout << "List find 10: " << (L.find(10) ? "YES" : "NO") << "\n\n";

    Stack S;
    S.push(5);
    S.push(7);
    int x = 0;
    S.pop(x);
    std::cout << "Stack pop: " << x << "\n";
    std::cout << "Stack empty: " << (S.isEmpty() ? "YES" : "NO") << "\n\n";

    Queue Q;
    Q.push_back(1);
    Q.push_back(2);
    Q.pop_front(x);
    std::cout << "Queue pop: " << x << "\n";
    std::cout << "Queue empty: " << (Q.isEmpty() ? "YES" : "NO") << "\n\n";

    // Сравнение
    int n1 = 10000;
    int n2 = 50000;

    std::cout << "Performance test (adding elements)\n";
    std::cout << "Threads = " << omp_get_max_threads() << "\n\n";

    // список
    {
        SinglyList a, b, c, d;

        long long t_seq_1 = add_list_seq(a, n1);
        long long t_par_1 = add_list_omp(b, n1);
        long long t_seq_2 = add_list_seq(c, n2);
        long long t_par_2 = add_list_omp(d, n2);

        std::cout << "Singly list:\n";
        std::cout << "N=" << n1 << " seq(us)=" << t_seq_1 << " omp(us)=" << t_par_1 << "\n";
        std::cout << "N=" << n2 << " seq(us)=" << t_seq_2 << " omp(us)=" << t_par_2 << "\n\n";
    }

    // очередь
    {
        Queue a, b, c, d;

        long long t_seq_1 = add_queue_seq(a, n1);
        long long t_par_1 = add_queue_omp(b, n1);
        long long t_seq_2 = add_queue_seq(c, n2);
        long long t_par_2 = add_queue_omp(d, n2);

        std::cout << "Queue:\n";
        std::cout << "N=" << n1 << " seq(us)=" << t_seq_1 << " omp(us)=" << t_par_1 << "\n";
        std::cout << "N=" << n2 << " seq(us)=" << t_seq_2 << " omp(us)=" << t_par_2 << "\n\n";
    }

    std::cout << "Done.\n";
    return 0;
}
