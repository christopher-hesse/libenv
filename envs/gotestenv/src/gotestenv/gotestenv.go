package main

// #cgo CFLAGS: -DNO_PROTOTYPE
/*
 #include "../../../../libenv/libenv.h"
*/
import "C"
import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

//export libenv_load
func libenv_load() {
	fmt.Println("libenv_load")
}

//export libenv_make
func libenv_make(num_envs C.uint32_t, options *C.struct_libenv_options) *C.libenv_venv {
	fmt.Println("libenv_make")
	return nil
}

//export libenv_get_spaces
func libenv_get_spaces(env *C.libenv_venv, name C.enum_libenv_spaces_name, out_spaces *C.struct_libenv_space) C.int {
	fmt.Println("libenv_get_spaces")
	if name == C.LIBENV_SPACES_ACTION {
		var space C.struct_libenv_space
		for i, c := range "action" {
			space.name[i] = C.char(c)
		}
		space._type = C.LIBENV_SPACE_TYPE_BOX
		space.dtype = C.LIBENV_DTYPE_UINT8
		space.ndim = 1

		space.shape = [16]C.int{1}
		binary.LittleEndian.PutUint32(space.low[:], 0)
		binary.LittleEndian.PutUint32(space.high[:], 16)

		var count C.int = 1
		if out_spaces != nil {
			// crazy method to convert a C pointer to a slice
			var slice_header = struct {
				addr uintptr
				len  int
				cap  int
			}{uintptr(unsafe.Pointer(out_spaces)), int(count), int(count)}
			out_spaces_slice := *(*[]C.struct_libenv_space)(unsafe.Pointer(&slice_header))
			out_spaces_slice[0] = space
		}
		return count
	}
	return 0
}

//export libenv_reset
func libenv_reset(env *C.libenv_venv, step *C.struct_libenv_step) {
	fmt.Println("libenv_reset")
}

//export libenv_step_async
func libenv_step_async(env *C.libenv_venv, acts *unsafe.Pointer, step *C.struct_libenv_step) {
	// fmt.Println("libenv_step")
}

//export libenv_step_wait
func libenv_step_wait(env *C.libenv_venv) {
	// fmt.Println("libenv_step")
}

//export libenv_close
func libenv_close(env *C.libenv_venv) {
	fmt.Println("libenv_close")
}

//export libenv_unload
func libenv_unload() {
	fmt.Println("libenv_unload")
}

func main() {}
