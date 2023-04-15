const sequence = ['Always Watching', 1500, 'Always Protecting', 1500];
            let index = 0;
            
            function animate() {
              const element = document.getElementById("animation");
              const current = sequence[index];
              if (typeof current === "string") {
                element.innerHTML = current;
              } else {
                setTimeout(animate, current);
              }
              index = (index + 1) % sequence.length;
            }
            
            function hover() {
              document.getElementById("startBTN").classList.add("hover");
            }
            
            function unhover() {
              document.getElementById("startBTN").classList.remove("hover");
            }
            
            function press() {
              document.getElementById("startBTN").classList.add("pressed");
            }
            
            function release() {
              document.getElementById("startBTN").classList.remove("pressed");
            }
            
            animate();