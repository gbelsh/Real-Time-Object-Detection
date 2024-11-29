# Real Time Object Detection/Tracking for Humans
## Step 1: Understanding the Problem

### What are we trying to do?
We need to create a system that can **detect people in real-time**, even in tough conditions like:
- **Low light** (dim or dark environments)
- **Partial occlusions** (where parts of the person are blocked by objects or other people)

This system will be used in situations like surveillance or monitoring, where it’s important to spot people quickly and accurately.

### Challenges We Face
1. **Low-Light Conditions**:
   - In low-light settings (like nighttime or poorly lit rooms), it's harder to spot people, as the details in the image become less clear.
   
2. **Occlusion**:
   - People might be partially hidden behind objects or other people, making it tough to identify them or track their movements.

3. **Real-Time Performance**:
   - The system needs to work **live** — it must process video or images quickly enough to catch people in motion without any delays.
   - It needs to detect people fast enough to work in a live setting, like security cameras or a robot navigating an environment.

### What Do We Need?
- **Accuracy**: 
   - The system must detect people accurately, even when it's dark or when they’re partially hidden. No false alarms, and no missing people!
   
- **Speed**:
   - It needs to work in real-time, meaning it should process frames fast enough to detect people on the go (about 30 frames per second or higher).
   
- **Hardware Considerations**:
   - Depending on where the system will be used, it must be efficient enough to run on the available hardware. If it's running on a camera, drone, or mobile device, the system must be optimized to run smoothly without draining too much power.

### Visuals We Could Use
- **Low-Light Example**: A picture or video showing a person in a dark room, where the lighting is minimal.
- **Occlusion Example**: A situation where one person is blocking part of another person, or objects obscure part of the figure, showing how tough it can be to detect someone in this scenario.

