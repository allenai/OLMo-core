// C++ extension backing OutputDiscardCheckpoint's storage-rebinding step.
//
// ``share_storage`` rebinds ``dst``'s underlying bytes to ``src``'s storage by
// mutating ``dst``'s existing ``StorageImpl`` in place (via ``set_data_ptr``),
// so any autograd-saved view of ``dst`` sees the new data through the same
// ``StorageImpl``. JIT-compiled at runtime via ``torch.utils.cpp_extension``;
// see ``output_discard_checkpoint.py`` for the Python fallback used when no
// C++ toolchain is available.

#include <torch/extension.h>

void share_storage(at::Tensor dst, at::Tensor src) {
    auto* dst_impl = dst.storage().unsafeGetStorageImpl();

    auto* src_storage_ref = new c10::Storage(src.storage());

    void*       data   = src_storage_ref->data_ptr().get();
    size_t      nbytes = src_storage_ref->nbytes();
    c10::Device device = src_storage_ref->device();

    c10::DataPtr shared(
        data,
        static_cast<void*>(src_storage_ref),
        [](void* ctx) { delete static_cast<c10::Storage*>(ctx); },
        device);

    dst_impl->set_data_ptr(std::move(shared));
    dst_impl->set_nbytes(nbytes);
}
