// import { useEffect, useRef } from "react"
// import { useSpring } from "framer-motion"

// export function useAnimatedText(target, transition) {
//   const ref = useRef(null)
//   const value = useSpring(target, transition)

//   useEffect(() => {
//     ref.current.innerText = target.toFixed(2)

//     return value.onChange((v) => {
//       ref.current.innerText = v.toFixed(2)
//     })
//   })
//   useEffect(() => value.set(target), [target])

//   return ref
// }

import { useEffect, useRef } from "react"
import { useSpring } from "framer-motion"

export function useAnimatedText(target, transition) {
  const ref = useRef(null)
  const value = useSpring(target, transition)

  useEffect(() => {
    if (ref.current) {
      ref.current.innerText = target.toFixed(2)
    }

    return value.onChange((v) => {
      if (ref.current) {
        ref.current.innerText = v.toFixed(2)
      }
    })
  }, [target, value])

  useEffect(() => value.set(target), [target, value])

  return ref
}

