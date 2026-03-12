# Project Tasks and Milestones

## Core question
How can one use a single scalar control input to bring a vibrating square membrane as close to rest as possible?

## Milestone 1: PDE to modal ODEs
- Write down the controlled wave equation on the unit square.
- State the Dirichlet eigenfunctions and eigenvalues.
- Derive the modal equations
  
  $$ \ddot{q}_{mn} + \omega_{mn}^2 q_{mn} = \beta_{mn} b(t). $$
- Explain the meaning of the coupling coefficient $` \beta_{mn} `$.

## Milestone 2: Build the finite-dimensional model
- Choose a truncation level $` M `$.
- Flatten the double indices $` (m,n) `$ into a vector of modal amplitudes.
- Construct the matrices $` A `$ and $` B `$ for the first-order system.
- Verify dimensions carefully.

## Milestone 3: Design the LQR controller
- Choose $` Q `$ and $` R `$.
- Solve the algebraic Riccati equation.
- Compute the feedback gain $` K `$.
- Explain how the choice of $` R `$ affects the controller.

## Milestone 4: Run simulations
- Choose at least one nontrivial initial condition.
- Simulate the open-loop response.
- Simulate the closed-loop response.
- Plot modal energy and control input.

## Milestone 5: Reconstruct the membrane
- Reconstruct $` u(x,y,t) `$ on a spatial grid from the modal coefficients.
- Plot snapshots of the membrane at several times.
- Create an animation or sequence of frames.

## Milestone 6: Investigate actuator placement
- Compare at least two actuator locations.
- Include the center $` \left( \frac{1}{2}, \frac{1}{2} \right) `$ as one test case.
- Identify modes with zero or very small coupling.
- Discuss how this affects stabilization.

## Minimum deliverables
- Short derivation of the modal model
- Clear definition of $` A `$, $` B `$, $` Q `$, $` R `$, $` K `$
- At least two plots from the simulation
- At least one membrane reconstruction figure
- Discussion of actuator placement

## Stretch ideas
- Replace the point actuator with a smooth Gaussian patch.
- Add light viscous damping and compare results.
- Track how performance changes as the truncation size grows.
- Estimate which low modes dominate the transient response.
