#pragma once

#include <cstddef>
#include <new>
#include <utility>

namespace FDTD_sycl {

template <class T>
class no_init_allocator {
public:
    using value_type = T;

    no_init_allocator() noexcept = default;

    template <class U>
    no_init_allocator(const no_init_allocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        return static_cast<T*>(::operator new(count * sizeof(T)));
    }

    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(static_cast<void*>(pointer));
    }

    template <class U>
    void construct(U*) noexcept {}

    template <class U, class A0, class... Args>
    void construct(U* pointer, A0&& first, Args&&... args) noexcept {
        ::new (pointer) U(std::forward<A0>(first), std::forward<Args>(args)...);
    }

    bool operator==(const no_init_allocator<T>&) const noexcept { return true; }
    bool operator!=(const no_init_allocator<T>&) const noexcept { return false; }
};

}
