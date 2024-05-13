function sysCall_init()
    -- ??????? ????? ??????? ? ??????????, ? ????? ??????? ???????
    local linkNames = {
        "/redundantRobot/link0",
        "/redundantRobot/link1",
        "/redundantRobot/link2",
        "/redundantRobot/link3",
        "/redundantRobot/link4",
        "/redundantRobot/link5"
    }
    local jointNames = {
        "/redundantRobot/joint0",
        "/redundantRobot/joint1",
        "/redundantRobot/joint2",
        "/redundantRobot/joint3",
        "/redundantRobot/joint4",
        "/redundantRobot/joint5",
        "/redundantRobot/joint6"
    }
    local newLengths = {1.5, 1.35, 1.4, 1.35, 1.3, 1.2}  -- ????? ????? ? ??????

    -- ????????? ???????? ???????
    for i = 1, #linkNames do
        local linkHandle = sim.getObject(linkNames[i])
        if linkHandle ~= -1 then
            if i == 4 or i == 5 then
                -- ???????? ????? ?????????????? ??????? ????? ??? X
                sim.scaleObject(linkHandle, newLengths[i], 1, 1)
            else
                -- ???????? ????? ???????????? ??????? ????? ??? Z
                sim.scaleObject(linkHandle, 1, 1, newLengths[i])
            end
            sim.addStatusbarMessage('Updated size of ' .. linkNames[i])
        else
            sim.addStatusbarMessage('Object ' .. linkNames[i] .. ' not found.')
        end
    end

    -- ?????????? ??????? ??????????
    local previousPosition = {0, 0, 0}
    for i = 1, #jointNames - 1 do
        local jointHandle = sim.getObject(jointNames[i])
        if jointHandle ~= -1 then
            local jointPosition = sim.getObjectPosition(jointHandle, -1)
            if not jointPosition then
                sim.addStatusbarMessage('Could not get position of ' .. jointNames[i])
                jointPosition = {0, 0, 0}
            end

            sim.addStatusbarMessage('Current position of ' .. jointNames[i] .. ': ' .. jointPosition[1] .. ', ' .. jointPosition[2] .. ', ' .. jointPosition[3])

            if i == 4 or i == 5 then
                jointPosition[1] = previousPosition[1] + newLengths[i]
            else
                jointPosition[3] = previousPosition[3] + newLengths[i]
            end
            sim.setObjectPosition(jointHandle, -1, jointPosition)
            previousPosition = jointPosition

            sim.addStatusbarMessage('Updated position of ' .. jointNames[i] .. ' to: ' .. jointPosition[1] .. ', ' .. jointPosition[2] .. ', ' .. jointPosition[3])
        else
            sim.addStatusbarMessage('Object ' .. jointNames[i] .. ' not found.')
        end
    end
end
