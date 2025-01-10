def ddpg_update(
    actor, critic, 
    actor_optimizer, critic_optimizer,
    replay_buffer, 
    batch_size=64, gamma=0.99
):
    # Sample batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones_tf = tf.convert_to_tensor(dones, dtype=tf.float32)
    
    # -----------------------
    # 1. Update Critic
    # -----------------------
    with tf.GradientTape() as tape:
        # Current Q(s,a)
        Q_vals = critic(states_tf, actions_tf)

        # Next a' = mu(next_state)
        next_actions = actor(next_states_tf)

        
        # Q(s', a')
        next_Q_vals = critic(next_states_tf, next_actions)
        
        
        # Target y = r + gamma*(1-done)* Q(s', a')
        y = tf.expand_dims(rewards_tf, -1) + gamma * (1.0 - tf.expand_dims(dones_tf, -1)) * next_Q_vals
        
        critic_loss = tf.reduce_mean((y - Q_vals)**2)
        
    
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    
    # -----------------------
    # 2. Update Actor
    # -----------------------
    with tf.GradientTape() as tape:
        # Actor tries to maximize Q(s, mu(s)) => minimize -Q()
        current_actions = actor(states_tf)
        
        actor_loss = -tf.reduce_mean(critic(states_tf, current_actions))
        
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    
    return critic_loss, actor_loss
